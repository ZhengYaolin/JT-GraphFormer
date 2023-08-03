import argparse
import pickle
import os

import numpy as np
from tqdm import tqdm


wo_k = "xsub"  # "xsub" or "xsub_wo_k"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='./gendata/ntu/xsub', required=False,
                        choices={'ntu/xsub', 'ntu/xview', 'ntu120/xsub', 'ntu120/xset'},
                        help='the work folder for storing results')
    parser.add_argument('--alpha', default=1, help='weighted summation', type=float)
    parser.add_argument('--joint_dir', default='./results_final/ntu60/'+wo_k+'/joint/',
                        help='Directory containing "score.pkl" for joint eval results')
    parser.add_argument('--bone_dir', default='./results_final/ntu60/'+wo_k+'/bone/',
                        help='Directory containing "score.pkl" for bone eval results')
    parser.add_argument('--joint_motion_dir', default='./results_final/ntu60/'+wo_k+'/joint_motion/')
    parser.add_argument('--bone_motion_dir', default='./results_final/ntu60/'+wo_k+'/bone_motion/')
    parser.add_argument('--detail_weight', default=True, help='use detail weight or not', type=bool)
    arg = parser.parse_args()

    dataset = arg.dataset
    if 'ntu/' in arg.dataset:
        if 'xsub' in arg.dataset:
            npz_data = np.load('./gendata/' + 'ntu/' + 'NTU60_XSub.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
        elif 'xview' in arg.dataset:
            npz_data = np.load('./gendata/' + 'ntu/' + 'NTU60_XView.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
    elif 'ntu120' in arg.dataset:
        if 'xsub' in arg.dataset:
            npz_data = np.load('./gendata/' + 'ntu120/' + 'NTU120_XSub.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
        elif 'xset' in arg.dataset:
            npz_data = np.load('./gendata/' + 'ntu120/' + 'NTU120_XSet.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
    else:
        raise NotImplementedError

    with open(os.path.join(arg.joint_dir + 'l2', 'score.pkl'), 'rb') as rjl2:
        rjl2 = list(pickle.load(rjl2).items())
    with open(os.path.join(arg.joint_dir + 'l4', 'score.pkl'), 'rb') as rjl4:
        rjl4 = list(pickle.load(rjl4).items())
    with open(os.path.join(arg.joint_dir + 'l6', 'score.pkl'), 'rb') as rjl6:
        rjl6 = list(pickle.load(rjl6).items())
    with open(os.path.join(arg.joint_dir + 'l8', 'score.pkl'), 'rb') as rjl8:
        rjl8 = list(pickle.load(rjl8).items())

    with open(os.path.join(arg.bone_dir + 'l2', 'score.pkl'), 'rb') as rbl2:
        rbl2 = list(pickle.load(rbl2).items())
    with open(os.path.join(arg.bone_dir + 'l4', 'score.pkl'), 'rb') as rbl4:
        rbl4 = list(pickle.load(rbl4).items())
    with open(os.path.join(arg.bone_dir + 'l6', 'score.pkl'), 'rb') as rbl6:
        rbl6 = list(pickle.load(rbl6).items())
    with open(os.path.join(arg.bone_dir + 'l8', 'score.pkl'), 'rb') as rbl8:
        rbl8 = list(pickle.load(rbl8).items())

    if arg.joint_motion_dir is not None:
        with open(os.path.join(arg.joint_motion_dir + 'l2', 'score.pkl'), 'rb') as rjml2:
            rjml2 = list(pickle.load(rjml2).items())
        with open(os.path.join(arg.joint_motion_dir + 'l4', 'score.pkl'), 'rb') as rjml4:
            rjml4 = list(pickle.load(rjml4).items())
        with open(os.path.join(arg.joint_motion_dir + 'l6', 'score.pkl'), 'rb') as rjml6:
            rjml6 = list(pickle.load(rjml6).items())
        with open(os.path.join(arg.joint_motion_dir + 'l8', 'score.pkl'), 'rb') as rjml8:
            rjml8 = list(pickle.load(rjml8).items())

    if arg.bone_motion_dir is not None:
        with open(os.path.join(arg.bone_motion_dir + 'l2', 'score.pkl'), 'rb') as rbml2:
            rbml2 = list(pickle.load(rbml2).items())
        with open(os.path.join(arg.bone_motion_dir + 'l4', 'score.pkl'), 'rb') as rbml4:
            rbml4 = list(pickle.load(rbml4).items())
        with open(os.path.join(arg.bone_motion_dir + 'l6', 'score.pkl'), 'rb') as rbml6:
            rbml6 = list(pickle.load(rbml6).items())
        with open(os.path.join(arg.bone_motion_dir + 'l8', 'score.pkl'), 'rb') as rbml8:
            rbml8 = list(pickle.load(rbml8).items())

    right_num = total_num = right_num_5 = 0
    max1 = max5 = acc = acc5 = 0

    # NTU60 Koopman
    arg.detail_weight = True
    alpha1 = [1, 1, 1, 1]
    alpha2 = [1, 1, 1, 1]
    alpha3 = [1, 1, 1, 1]
    alpha4 = [1, 1, 1, 1]
    if not arg.detail_weight:
        arg.alpha = [1.6, 1.5, 1, 1]  # koopman  4s 93.3
    else:
        arg.alpha = [1.1, 1.5, 0.9, 1]  # koopman 4s 93.4
        alpha1 = [1.1, 1, 1.1, 1]
        alpha2 = [1, 0.95, 1.22, 1]
        alpha3 = [1, 1.1, 1.2, 1.1]
        alpha4 = [0.7, 1, 1, 0.9]


    for i in tqdm(range(len(label))):
        l = label[i]
        _, rj1 = rjl2[i]
        _, rj2 = rjl4[i]
        _, rj3 = rjl6[i]
        _, rj4 = rjl8[i]
        r11 = alpha1[0] * rj1 + alpha1[1] * rj2 + alpha1[2] * rj3 + alpha1[3] * rj4

        _, rb1 = rbl2[i]
        _, rb2 = rbl4[i]
        _, rb3 = rbl6[i]
        _, rb4 = rbl8[i]
        r22 = alpha2[0] * rb1 + alpha2[1] * rb2 + alpha2[2] * rb3 + alpha2[3] * rb4

        _, rjm1 = rjml2[i]
        _, rjm2 = rjml4[i]
        _, rjm3 = rjml6[i]
        _, rjm4 = rjml8[i]
        r33 = alpha3[0] * rjm1 + alpha3[1] * rjm2 + alpha3[2] * rjm3 + alpha3[3] * rjm4

        _, rbm1 = rbml2[i]
        _, rbm2 = rbml4[i]
        _, rbm3 = rbml6[i]
        _, rbm4 = rbml8[i]
        r44 = alpha4[0] * rbm1 + alpha4[1] * rbm2 + alpha4[2] * rbm3 + alpha4[3] * rbm4

        r = r11 * arg.alpha[0] + r22 * arg.alpha[1] + r33 * arg.alpha[2] + r44 * arg.alpha[3]
        rank_5 = r.argsort()[-5:]
        right_num_5 += int(int(l) in rank_5)
        r = np.argmax(r)
        right_num += int(r == int(l))
        total_num += 1
    acc = right_num / total_num
    acc5 = right_num_5 / total_num

    print('Top1 Acc: {:.1f}%'.format(acc * 100))
    print('Top5 Acc: {:.1f}%'.format(acc5 * 100))
