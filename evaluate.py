import numpy as np
from util import write
import os
import torch
from data_loader import get_test_loader
from torch.autograd import Variable
import torch.nn.functional as F
from pretrain import Identify_net
import torch.backends.cudnn as cudnn
from model_reid import DReid

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

project_path = '/home/wxt/PRCGAN'

# DATASET = project_path + '/market'
# TEST = os.path.join(DATASET, 'test')
# TEST_NUM = 19732
# TRAIN = os.path.join(DATASET, 'train')
# TRAIN_NUM = 12936
# QUERY = os.path.join(DATASET, 'probe')
# QUERY_NUM = 3368

'''
DATASET = project_path + '../dataset/Duke'
TEST = os.path.join(DATASET, 'bounding_box_test')
TEST_NUM = 17661
QUERY = os.path.join(DATASET, 'query')
QUERY_NUM = 2228
'''

'''
DATASET = project_path + '/dataset/CUHK03'
TEST = os.path.join(DATASET, 'bbox_test')
TEST_NUM = 5332
QUERY = os.path.join(DATASET, 'query')
QUERY_NUM = 1400
'''

DATASET = project_path + '/grid'
TEST = os.path.join(DATASET, 'gallery')
TEST_NUM = 1025
TRAIN = os.path.join(DATASET, 'train')
TRAIN_NUM = 500
QUERY = os.path.join(DATASET, 'probe')
QUERY_NUM = 250


"""
load pretrain model
"""
# source_model_path = './market_softmax_pretrain.pkl'
# net = Identify_net(class_count=751)
# net = torch.nn.DataParallel(net).cuda()
# cudnn.benchmark = True
# net.load_state_dict(torch.load(source_model_path))
# print('pretrain model loading success.')

# source_model_path = './models/dreid-80000.pkl'
# net = DReid(class_count=751)
# net.cuda()
# net.load_state_dict(torch.load(source_model_path))
# print('pretrain model loading success.')


def extract_feature(dir_path, net):
    features = []
    infos = []
    # get test dataset
    test_loader = get_test_loader(dir_path)
    # change mode to test
    net.eval()

    for i, (images, info) in enumerate(test_loader):
        if torch.cuda.is_available():
            images = Variable(images, volatile=True).cuda()
        else:
            images = Variable(images, volatile=True)

        feature, output = net(images)
        feature = feature.cpu()
        feature = feature.data.numpy()
        feature = np.squeeze(feature)
        features.append(feature)
        # features.append(np.squeeze(feature))
        person = int(np.squeeze(info[0].numpy()))
        camera = int(np.squeeze(info[1].numpy()))
        info = (person, camera)
        infos.append(info)
    features = np.asarray(features)

    return features, infos


def similarity_matrix(query_f, test_f):
    query_f = torch.FloatTensor(query_f)
    test_f = torch.FloatTensor(test_f)

    query_t_norm = F.normalize(query_f, p=2, dim=1)
    test_t_norm = F.normalize(test_f, p=2, dim=1)

    test_t_norm = torch.t(test_t_norm)
    tensor = torch.mm(query_t_norm, test_t_norm)
    tensor = tensor.numpy()
    print(tensor.shape)
    # descend
    return tensor


def sort_similarity(query_f, test_f):
    result = similarity_matrix(query_f, test_f)
    result_argsort = np.argsort(-result, axis=1)
    return result, result_argsort


def test_predict(net, probe_path, gallery_path, pid_path, score_path):
    test_f, test_info = extract_feature(gallery_path, net)
    query_f, query_info = extract_feature(probe_path, net)
    result, result_argsort = sort_similarity(query_f, test_f)
    for i in range(len(result)):
        result[i] = result[i][result_argsort[i]]
    result = np.array(result)
    # ignore top1 because it's the origin image
    np.savetxt(pid_path, result_argsort, fmt='%d')
    np.savetxt(score_path, result, fmt='%.4f')
    # map_rank_eval(query_info, test_info, result_argsort)


def market_result_eval(predict_path, log_path='market_eval_0.log'):
    res = np.genfromtxt(predict_path, delimiter=' ')
    print('predict info get, extract gallery info start')
    test_info = extract_info(TEST)
    print('extract probe info start')
    query_info = extract_info(QUERY)
    print('start evaluate map and rank acc')
    rank1, mAP = map_rank_quick_eval(query_info, test_info, res)
    write(log_path, predict_path + '\n')
    write(log_path, '%f\t%f\n' % (rank1, mAP))


def grid_result_eval(predict_path, log_path='grid_eval.log'):
    pids4probes = np.genfromtxt(predict_path, delimiter=' ')
    probe_shoot = [0, 0, 0, 0, 0]
    for i, pids in enumerate(pids4probes):
        for j, pid in enumerate(pids):
            if pid - i == 775:
                if j == 0:
                    for k in range(5):
                        probe_shoot[k] += 1
                elif j < 5:
                    for k in range(1,5):
                        probe_shoot[k] += 1
                elif j < 10:
                    for k in range(2,5):
                        probe_shoot[k] += 1
                elif j < 20:
                    for k in range(3,5):
                        probe_shoot[k] += 1
                elif j < 50:
                    for k in range(4,5):
                        probe_shoot[k] += 1
                break
    probe_acc = [shoot/len(pids4probes) for shoot in probe_shoot]
    write(log_path, predict_path + '\n')
    write(log_path, '%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n' % (probe_acc[0], probe_acc[1], probe_acc[2], probe_acc[3], probe_acc[4]))
    print(predict_path)
    print(probe_acc)


def grid_result_eval_idt(predict_path, log_path='grid_eval_idt.log'):
    pids4probes = np.genfromtxt(predict_path, delimiter=' ')
    probe_shoot = [0, 0, 0, 0, 0]
    for i, pids in enumerate(pids4probes):
        for j, pid in enumerate(pids):
            if pid - i == 775:
                if j == 0:
                    for k in range(5):
                        probe_shoot[k] += 1
                elif j < 5:
                    for k in range(1,5):
                        probe_shoot[k] += 1
                elif j < 10:
                    for k in range(2,5):
                        probe_shoot[k] += 1
                elif j < 20:
                    for k in range(3,5):
                        probe_shoot[k] += 1
                elif j < 50:
                    for k in range(4,5):
                        probe_shoot[k] += 1
                break
    probe_acc = [shoot/len(pids4probes) for shoot in probe_shoot]
    write(log_path, predict_path + '\n')
    write(log_path, '%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n' % (probe_acc[0], probe_acc[1], probe_acc[2], probe_acc[3], probe_acc[4]))
    print(predict_path)
    print(probe_acc)


def extract_info(dir_path):
    infos = []
    for image_name in sorted(os.listdir(dir_path)):
        if '.txt' in image_name or '.db' in image_name:
            continue
        arr = image_name.split('_')
        person = int(arr[0])
        camera = int(arr[1][1])
        infos.append((person, camera))

    return infos


def map_rank_quick_eval(query_info, test_info, result_argsort):
    # about 10% lower than matlab result
    # for evaluate rank1 and map
    match = []
    junk = []

    for q_index, (qp, qc) in enumerate(query_info):
        tmp_match = []
        tmp_junk = []
        for t_index in range(len(test_info)):
            p_t_idx = result_argsort[q_index][t_index]
            p_info = test_info[int(p_t_idx)]

            tp = p_info[0]
            tc = p_info[1]
            if tp == qp and qc != tc:
                tmp_match.append(t_index)
            elif tp == qp or tp == -1:
                tmp_junk.append(t_index)
        match.append(tmp_match)
        junk.append(tmp_junk)

    rank_1 = 0.0
    mAP = 0.0
    rank1_list = list()
    for idx in range(len(query_info)):
        if idx % 100 == 0:
            print('evaluate img %d' % idx)
        recall = 0.0
        precision = 1.0
        ap = 0.0
        YES = match[idx]
        IGNORE = junk[idx]
        ig_cnt = 0
        for ig in IGNORE:
            if ig < YES[0]:
                ig_cnt += 1
            else:
                break
        if ig_cnt >= YES[0]:
            rank_1 += 1
            rank1_list.append(1)
        else:
            rank1_list.append(0)

        for i, k in enumerate(YES):
            ig_cnt = 0
            for ig in IGNORE:
                if ig < k:
                    ig_cnt += 1
                else:
                    break
            cnt = k + 1 - ig_cnt
            hit = i + 1
            tmp_recall = hit / len(YES)
            tmp_precision = hit / cnt
            ap = ap + (tmp_recall - recall) * ((precision + tmp_precision) / 2)
            recall = tmp_recall
            precision = tmp_precision

        mAP += ap
    rank1_acc = rank_1 / QUERY_NUM
    mAP = mAP / QUERY_NUM
    print('Rank 1:\t%f' % rank1_acc)
    print('mAP:\t%f' % mAP)
    np.savetxt('rank_1.log', np.array(rank1_list), fmt='%d')
    return rank1_acc, mAP


# if __name__ == '__main__':
    # test_predict(net, QUERY, TEST,
    #              pid_path='/home/xintong/Desktop/ReidCycleGAN/result/market_pid.txt',
    #              score_path='/home/xintong/Desktop/ReidCycleGAN/result/market_score.txt')
    #
    # market_result_eval('/home/xintong/Desktop/ReidCycleGAN/result/market_pid.txt')

    # market_result_eval('/home/xintong/Desktop/ReidCycleGAN/result/cross_filter_pid.log')

def evaluate():
    source_model_path = './market_softmax_pretrain.pkl'
    net = Identify_net(class_count=751)
    net = torch.nn.DataParallel(net).cuda()
    cudnn.benchmark = True
    net.load_state_dict(torch.load(source_model_path))
    test_predict(net, QUERY, TEST,
                 pid_path='/home/wxt/PRCGAN/result/grid_pid.txt',
                 score_path='/home/wxt/PRCGAN/result/grid_score.txt')

    grid_result_eval('/home/wxt/PRCGAN/result/grid_pid.txt')


def evaluate_idt(dreid_path):
    idt_model_path = dreid_path
    idt_net = DReid(class_count=751)
    idt_net.cuda()
    idt_net.load_state_dict(torch.load(idt_model_path))
    test_predict(idt_net, QUERY, TEST,
                 pid_path='/home/wxt/PRCGAN/result/grid_pid.txt',
                 score_path='/home/wxt/PRCGAN/result/grid_score.txt')

    grid_result_eval_idt('/home/wxt/PRCGAN/result/grid_pid.txt')
