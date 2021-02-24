import math, shutil, os, time, argparse, json, re, sys
import numpy as np
import scipy.io as sio
from PIL import Image
import multiprocessing
from multiprocessing import Queue
from operator import itemgetter
import pprint as pp

'''
Prepares the GazeCapture dataset for use with the pytorch code. Crops images, compiles JSONs into metadata.mat

Author: Petr Kellnhofer ( pkel_lnho (at) gmai_l.com // remove underscores and spaces), 2018. 

Website: http://gazecapture.csail.mit.edu/

Cite:

Eye Tracking for Everyone
K.Krafka*, A. Khosla*, P. Kellnhofer, H. Kannan, S. Bhandarkar, W. Matusik and A. Torralba
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016

@inproceedings{cvpr2016_gazecapture,
Author = {Kyle Krafka and Aditya Khosla and Petr Kellnhofer and Harini Kannan and Suchendra Bhandarkar and Wojciech Matusik and Antonio Torralba},
Title = {Eye Tracking for Everyone},
Year = {2016},
Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}
}

'''

parser = argparse.ArgumentParser(description='iTracker-pytorch-PrepareDataset.')
parser.add_argument('--dataset_path', help="Path to extracted files. It should have folders called '%%05d' in it.")
parser.add_argument('--output_path', default=None, help="Where to write the output. Can be the same as dataset_path if you wish (=default).")
args = parser.parse_args()

g_meta_queue = Queue()
g_meta_list = []
g_meta = {
        'labelRecNum': [],
        'frameIndex': [],
        'labelDotXCam': [],
        'labelDotYCam': [],
        'labelFaceGrid': [],
    }

def process_recording(recordings, thread_id):
    # Output structure
    meta = {
        'labelRecNum': [],
        'frameIndex': [],
        'labelDotXCam': [],
        'labelDotYCam': [],
        'labelFaceGrid': [],
    }

    for i,recording in enumerate(recordings):
        print('[%d/%d] Thread %d Processing recording %s (%.2f%%)' % (i, len(recordings), thread_id, recording, i / len(recordings) * 100))
        recDir = os.path.join(args.dataset_path, recording)
        recDirOut = os.path.join(args.output_path, recording)

        # Read JSONs
        appleFace = readJson(os.path.join(recDir, 'appleFace.json'))
        if appleFace is None:
            continue
        appleLeftEye = readJson(os.path.join(recDir, 'appleLeftEye.json'))
        if appleLeftEye is None:
            continue
        appleRightEye = readJson(os.path.join(recDir, 'appleRightEye.json'))
        if appleRightEye is None:
            continue
        dotInfo = readJson(os.path.join(recDir, 'dotInfo.json'))
        if dotInfo is None:
            continue
        faceGrid = readJson(os.path.join(recDir, 'faceGrid.json'))
        if faceGrid is None:
            continue
        frames = readJson(os.path.join(recDir, 'frames.json'))
        if frames is None:
            continue
        # info = readJson(os.path.join(recDir, 'info.json'))
        # if info is None:
        #     continue
        # screen = readJson(os.path.join(recDir, 'screen.json'))
        # if screen is None:
        #     continue

        facePath = preparePath(os.path.join(recDirOut, 'appleFace'))
        leftEyePath = preparePath(os.path.join(recDirOut, 'appleLeftEye'))
        rightEyePath = preparePath(os.path.join(recDirOut, 'appleRightEye'))

        # Preprocess
        allValid = np.logical_and(np.logical_and(appleFace['IsValid'], appleLeftEye['IsValid']), np.logical_and(appleRightEye['IsValid'], faceGrid['IsValid']))
        if not np.any(allValid):
            continue

        frames = np.array([int(re.match('(\d{5})\.jpg$', x).group(1)) for x in frames])

        bboxFromJson = lambda data: np.stack((data['X'], data['Y'], data['W'],data['H']), axis=1).astype(int)
        faceBbox = bboxFromJson(appleFace) + [-1,-1,1,1] # for compatibility with matlab code
        leftEyeBbox = bboxFromJson(appleLeftEye) + [0,-1,0,0]
        rightEyeBbox = bboxFromJson(appleRightEye) + [0,-1,0,0]
        leftEyeBbox[:,:2] += faceBbox[:,:2] # relative to face
        rightEyeBbox[:,:2] += faceBbox[:,:2]
        faceGridBbox = bboxFromJson(faceGrid)


        for j,frame in enumerate(frames):
            # Can we use it?
            if not allValid[j]:
                continue

            # Load image
            imgFile = os.path.join(recDir, 'frames', '%05d.jpg' % frame)
            if not os.path.isfile(imgFile):
                logError('Warning: Could not read image file %s!' % imgFile)
                continue
            img = Image.open(imgFile)        
            if img is None:
                logError('Warning: Could not read image file %s!' % imgFile)
                continue
            img = np.array(img.convert('RGB'))

            # Crop images
            imFace = cropImage(img, faceBbox[j,:])
            imEyeL = cropImage(img, leftEyeBbox[j,:])
            imEyeR = cropImage(img, rightEyeBbox[j,:])

            # Save images
            Image.fromarray(imFace).save(os.path.join(facePath, '%05d.jpg' % frame), quality=95)
            Image.fromarray(imEyeL).save(os.path.join(leftEyePath, '%05d.jpg' % frame), quality=95)
            Image.fromarray(imEyeR).save(os.path.join(rightEyePath, '%05d.jpg' % frame), quality=95)

            # Collect metadata
            meta['labelRecNum'] += [int(recording)]
            meta['frameIndex'] += [frame]
            meta['labelDotXCam'] += [dotInfo['XCam'][j]]
            meta['labelDotYCam'] += [dotInfo['YCam'][j]]
            meta['labelFaceGrid'] += [faceGridBbox[j,:]]

    return meta

def run_process(thread_id, name, recordings):
    print("Starting " + name)

    meta = process_recording(recordings, thread_id)
    meta_tup = (thread_id, meta) # Tuple so we can sort later on

    # Add to global thread-safe queue
    g_meta_queue.put(meta_tup)

    print("{} finished. Processed {} recordings".format(name, len(recordings)))

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def main():
    if args.output_path is None:
        args.output_path = args.dataset_path
    
    if args.dataset_path is None or not os.path.isdir(args.dataset_path):
        raise RuntimeError('No such dataset folder %s!' % args.dataset_path)

    preparePath(args.output_path)

    # list recordings
    recordings = os.listdir(args.dataset_path)
    recordings = np.array(recordings, np.object)
    recordings = recordings[[os.path.isdir(os.path.join(args.dataset_path, r)) for r in recordings]]
    recordings.sort()

    NUM_THREADS = 15
#     num_recordings = len(recordings)
    # Max number of recordings a thread can have. 
    # Thus, (N-1) threads will have M recordings each. The last thread will have the remainder.
#     max_recordings_per_thread = math.ceil(num_recordings/NUM_THREADS)
    
    # Split recordings into approximately equal sized chunks for each thread
    chunked_recordings = list(split(recordings, NUM_THREADS))
    processes = []
    pp.pprint(chunked_recordings)
    num_processes = 0

    for i,recording in enumerate(chunked_recordings):
        # Start parallel processes
        name = "Thread " + str(i)
        p = multiprocessing.Process(target=run_process, args=(i, name, recording))
        processes.append((i, p))
        p.start()
        num_processes += 1

    meta_list = []
    num_processes_remaining = int(num_processes)
    
    while num_processes_remaining > 0:
        while not g_meta_queue.empty():
            meta_list.append(g_meta_queue.get())
            num_processes_remaining -= 1
            print("{} processes remaining".format(num_processes_remaining))
            
        time.sleep(5)
        
    for p_tup in processes:
        p_id, p = p_tup
        # Join processes
        p.join()
        print("Joined process {}".format(p_id))

    # Sort meta_list in order of thread id (so lower thread num comes first)
    meta_list.sort(key=itemgetter(0))

    for item in meta_list:
        thread_id, meta = item

        for key in meta:
            g_meta[key] += meta[key]
    
    print("Created g_meta database")
    
    # Integrate
    g_meta['labelRecNum'] = np.stack(g_meta['labelRecNum'], axis = 0).astype(np.int16)
    g_meta['frameIndex'] = np.stack(g_meta['frameIndex'], axis = 0).astype(np.int32)
    g_meta['labelDotXCam'] = np.stack(g_meta['labelDotXCam'], axis = 0)
    g_meta['labelDotYCam'] = np.stack(g_meta['labelDotYCam'], axis = 0)
    g_meta['labelFaceGrid'] = np.stack(g_meta['labelFaceGrid'], axis = 0).astype(np.uint8)

    # Load reference metadata
    print('Will compare to the reference GitHub dataset metadata.mat...')
    reference = sio.loadmat('./reference_metadata.mat', struct_as_record=False) 
    reference['labelRecNum'] = reference['labelRecNum'].flatten()
    reference['frameIndex'] = reference['frameIndex'].flatten()
    reference['labelDotXCam'] = reference['labelDotXCam'].flatten()
    reference['labelDotYCam'] = reference['labelDotYCam'].flatten()
    reference['labelTrain'] = reference['labelTrain'].flatten()
    reference['labelVal'] = reference['labelVal'].flatten()
    reference['labelTest'] = reference['labelTest'].flatten()

    # Find mapping
    mKey = np.array(['%05d_%05d' % (rec, frame) for rec, frame in zip(g_meta['labelRecNum'], g_meta['frameIndex'])], np.object)
    rKey = np.array(['%05d_%05d' % (rec, frame) for rec, frame in zip(reference['labelRecNum'], reference['frameIndex'])], np.object)
    mIndex = {k: i for i,k in enumerate(mKey)}
    rIndex = {k: i for i,k in enumerate(rKey)}
    mToR = np.zeros((len(mKey,)),int) - 1
    for i,k in enumerate(mKey):
        if k in rIndex:
            mToR[i] = rIndex[k]
        else:
            logError('Did not find rec_frame %s from the new dataset in the reference dataset!' % k)
    rToM = np.zeros((len(rKey,)),int) - 1
    for i,k in enumerate(rKey):
        if k in mIndex:
            rToM[i] = mIndex[k]
        else:
            logError('Did not find rec_frame %s from the reference dataset in the new dataset!' % k, critical = False)
            #break

    # Copy split from reference
    g_meta['labelTrain'] = np.zeros((len(g_meta['labelRecNum'],)),np.bool)
    g_meta['labelVal'] = np.ones((len(g_meta['labelRecNum'],)),np.bool) # default choice
    g_meta['labelTest'] = np.zeros((len(g_meta['labelRecNum'],)),np.bool)

    validMappingMask = mToR >= 0
    g_meta['labelTrain'][validMappingMask] = reference['labelTrain'][mToR[validMappingMask]]
    g_meta['labelVal'][validMappingMask] = reference['labelVal'][mToR[validMappingMask]]
    g_meta['labelTest'][validMappingMask] = reference['labelTest'][mToR[validMappingMask]]

    # Write out metadata
    metaFile = os.path.join(args.output_path, 'metadata.mat')
    print('Writing out the metadata.mat to %s...' % metaFile)
    sio.savemat(metaFile, g_meta)
    
    # Statistics
    nMissing = np.sum(rToM < 0)
    nExtra = np.sum(mToR < 0)
    totalMatch = len(mKey) == len(rKey) and np.all(np.equal(mKey, rKey))
    print('======================\n\tSummary\n======================')    
    print('Total added %d frames from %d recordings.' % (len(g_meta['frameIndex']), len(np.unique(g_meta['labelRecNum']))))
    if nMissing > 0:
        print('There are %d frames missing in the new dataset. This may affect the results. Check the log to see which files are missing.' % nMissing)
    else:
        print('There are no missing files.')
    if nExtra > 0:
        print('There are %d extra frames in the new dataset. This is generally ok as they were marked for validation split only.' % nExtra)
    else:
        print('There are no extra files that were not in the reference dataset.')
    if totalMatch:
        print('The new metadata.mat is an exact match to the reference from GitHub (including ordering)')

    #import pdb; pdb.set_trace()
    input("Press Enter to continue...")




def readJson(filename):
    if not os.path.isfile(filename):
        logError('Warning: No such file %s!' % filename)
        return None

    with open(filename) as f:
        try:
            data = json.load(f)
        except:
            data = None

    if data is None:
        logError('Warning: Could not read file %s!' % filename)
        return None

    return data

def preparePath(path, clear = False):
    if not os.path.isdir(path):
        os.makedirs(path, 0o777)
    if clear:
        files = os.listdir(path)
        for f in files:
            fPath = os.path.join(path, f)
            if os.path.isdir(fPath):
                shutil.rmtree(fPath)
            else:
                os.remove(fPath)

    return path

def logError(msg, critical = False):
    print(msg)
    if critical:
        sys.exit(1)


def cropImage(img, bbox):
    bbox = np.array(bbox, int)

    aSrc = np.maximum(bbox[:2], 0)
    bSrc = np.minimum(bbox[:2] + bbox[2:], (img.shape[1], img.shape[0]))

    aDst = aSrc - bbox[:2]
    bDst = aDst + (bSrc - aSrc)

    res = np.zeros((bbox[3], bbox[2], img.shape[2]), img.dtype)    
    res[aDst[1]:bDst[1],aDst[0]:bDst[0],:] = img[aSrc[1]:bSrc[1],aSrc[0]:bSrc[0],:]

    return res


if __name__ == "__main__":
    main()
    print('DONE')
