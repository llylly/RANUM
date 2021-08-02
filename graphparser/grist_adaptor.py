"""
    The adaptor for GRIST benchmark
    GRIST: https://github.com/Jacob-yen/GRIST

    TODO:
        - update the case_dict to reflect the real directory
        - global change the data path for the benchmark
        - dynamically detect whether it is tf or pytorch model
        - if it is tf model, get the statement after initialize_all_variables and conduct conversion
        - if it is pytorch model, warning the existence and then I try to transform it manually...
"""


current_selected_scripts = {
    "ID_1": "model_zoo/grist_models/ID_1/ghips1_grist.py",
    "ID_2": "model_zoo/grist_models/ID_2/command_line/run_logistic_regression_grist.py",
    "ID_3": "model_zoo/grist_models/ID_3/ghips9_grist.py",
    "ID_4": "model_zoo/grist_models/ID_4/test/nn/models/test_autoencoder_grist.py",
    "ID_5": "model_zoo/grist_models/ID_5/tests/test_losses_grist.py",
    "ID_6": "model_zoo/grist_models/ID_6/soips1_grist.py",
    "ID_7": "model_zoo/grist_models/ID_7/soips2_grist.py",
    "ID_8": "model_zoo/grist_models/ID_8/soips6_grist.py",
    "ID_9": "model_zoo/grist_models/ID_9/soips7_grist.py",
    "ID_10": "model_zoo/grist_models/ID_10/soips14_grist.py",
    "ID_11": "model_zoo/grist_models/ID_11/nan_model_truediv_grist.py",
    "ID_12": "model_zoo/grist_models/ID_12/main_grist.py",
    "ID_13": "model_zoo/grist_models/ID_13/test/utils/test_softmax_grist.py",
    "ID_14": "model_zoo/grist_models/ID_14/v3/train_grist.py",
    "ID_15": "model_zoo/grist_models/ID_15/train_grist.py",
    "ID_16": "model_zoo/grist_models/ID_16/nan_model_grist.py",
    "ID_17": "model_zoo/grist_models/ID_17/test_toy_grist.py",
    "ID_18": "model_zoo/grist_models/ID_18/code_10_image_grist.py",
    "ID_19": "model_zoo/grist_models/ID_19/mnist_grist.py",
    "ID_20": "model_zoo/grist_models/ID_20/logistic_regression_grist.py",
    "ID_21": "model_zoo/grist_models/ID_21/mnist_softmax_grist.py",
    "ID_22": "model_zoo/grist_models/ID_22/My_pytorch1_grist.py",
    "ID_23": "model_zoo/grist_models/ID_23/ch5.3_softmax_grist.py",
    "ID_24": "model_zoo/grist_models/ID_24/Mnist_grist.py",
    "ID_25": "model_zoo/grist_models/ID_25/mnist_grist.py",
    "ID_26": "model_zoo/grist_models/ID_26/0401_logistic_regression_grist.py",
    "ID_27": "model_zoo/grist_models/ID_27/0503_softmax_regression_cost_grist.py",
    "ID_28": "model_zoo/grist_models/ID_28/sc_train_l2reg_div2_grist.py",
    "ID_29": "model_zoo/grist_models/ID_29/temp_grist.py",
    "ID_30": "model_zoo/grist_models/ID_30/softmax_grist.py",
    "ID_31": "model_zoo/grist_models/ID_31/GAN_MNIST_grist.py",
    "ID_32": "model_zoo/grist_models/ID_32/0504_softmax_regression_grist.py",
    "ID_33": "model_zoo/grist_models/ID_33/logistic_classification_grist.py",
    "ID_34": "model_zoo/grist_models/ID_34/softmax_classification_grist.py",
    "ID_35": "model_zoo/grist_models/ID_35/ch10_04_01_grist.py",
    "ID_36": "model_zoo/grist_models/ID_36/ch10_04_03_Pic_10_05_grist.py",
    "ID_37": "model_zoo/grist_models/ID_37/lab-05-1-logistic_regression_grist.py",
    "ID_38": "model_zoo/grist_models/ID_38/lab-09-1-xor_grist.py",
    "ID_39": "model_zoo/grist_models/ID_39/ch10_04_04_Pic_10_06_grist.py",
    "ID_40": "model_zoo/grist_models/ID_40/src/ch5/bmi_grist.py",
    "ID_41": "model_zoo/grist_models/ID_41/src/ch5/tb-bmi_grist.py",
    "ID_42": "model_zoo/grist_models/ID_42/src/ch5/tb-bmi2_grist.py",
    "ID_43": "model_zoo/grist_models/ID_43/ch10_04_05_Pic_10_07_grist.py",
    "ID_44": "model_zoo/grist_models/ID_44/examples/cnn-mnist/mnist_example_grist.py",
    "ID_45": "model_zoo/grist_models/ID_45/MNIST_Digit_classification_using_CNN_grist.py",
    "ID_46": "model_zoo/grist_models/ID_46/torch20_softmax_grist.py",
    "ID_47": "model_zoo/grist_models/ID_47/torch21_Softmax_Reg1_grist.py",
    "ID_48": "model_zoo/grist_models/ID_48/day01_grist.py",
    "ID_49": "model_zoo/grist_models/ID_49/ch10_04_06_Pic_10_08_grist.py",
    "ID_50": "model_zoo/grist_models/ID_50/DL_1L_grist.py",
    "ID_51": "model_zoo/grist_models/ID_51/rbm_grist.py",
    "ID_52": "model_zoo/grist_models/ID_52/5_mnist_full_connection_grist.py",
    "ID_53": "model_zoo/grist_models/ID_53/h02_20191644_grist.py",
    "ID_54": "model_zoo/grist_models/ID_54/h03_20191644_grist.py",
    "ID_55": "model_zoo/grist_models/ID_55/logistic_regression_grist.py",
    "ID_56": "model_zoo/grist_models/ID_56/01_Logistic_Regression_grist.py",
    "ID_57": "model_zoo/grist_models/ID_57/03_softmax_cost_grist.py",
    "ID_58": "model_zoo/grist_models/ID_58/test_grist.py",
    "ID_59": "model_zoo/grist_models/ID_59/code_06_grist.py",
    "ID_60": "model_zoo/grist_models/ID_60/simple_saveAndReloadModel_grist.py",
    "ID_61": "model_zoo/grist_models/ID_61/multi_layer_perceptron_grist.py",
    "ID_62": "model_zoo/grist_models/ID_62/denoising_RBM_grist.py",
    "ID_63": "model_zoo/grist_models/ID_63/main_grist.py"
}

current_new_scripts = {
    "ID_1": "model_zoo/grist_models/ID_1/ghips1_DEBARUS.py",
    "ID_2": "model_zoo/grist_models/ID_2/command_line/run_logistic_regression_DEBARUS.py",
    "ID_3": "model_zoo/grist_models/ID_3/ghips9_DEBARUS.py",
    "ID_4": "model_zoo/grist_models/ID_4/test/nn/models/test_autoencoder_DEBARUS.py",
    "ID_5": "model_zoo/grist_models/ID_5/tests/test_losses_DEBARUS.py",
    "ID_6": "model_zoo/grist_models/ID_6/soips1_DEBARUS.py",
    "ID_7": "model_zoo/grist_models/ID_7/soips2_DEBARUS.py",
    "ID_8": "model_zoo/grist_models/ID_8/soips6_DEBARUS.py",
    "ID_9": "model_zoo/grist_models/ID_9/soips7_DEBARUS.py",
    "ID_10": "model_zoo/grist_models/ID_10/soips14_DEBARUS.py",
    "ID_11": "model_zoo/grist_models/ID_11/nan_model_truediv_DEBARUS.py",
    "ID_12": "model_zoo/grist_models/ID_12/main_DEBARUS.py",
    "ID_13": "model_zoo/grist_models/ID_13/test/utils/test_softmax_DEBARUS.py",
    "ID_14": "model_zoo/grist_models/ID_14/v3/train_DEBARUS.py",
    "ID_15": "model_zoo/grist_models/ID_15/train_DEBARUS.py",
    "ID_16": "model_zoo/grist_models/ID_16/nan_model_DEBARUS.py",
    "ID_17": "model_zoo/grist_models/ID_17/test_toy_DEBARUS.py",
    "ID_18": "model_zoo/grist_models/ID_18/code_10_image_DEBARUS.py",
    "ID_19": "model_zoo/grist_models/ID_19/mnist_DEBARUS.py",
    "ID_20": "model_zoo/grist_models/ID_20/logistic_regression_DEBARUS.py",
    "ID_21": "model_zoo/grist_models/ID_21/mnist_softmax_DEBARUS.py",
    "ID_22": "model_zoo/grist_models/ID_22/My_pytorch1_DEBARUS.py",
    "ID_23": "model_zoo/grist_models/ID_23/ch5_DEBARUS.3_softmax_DEBARUS.py",
    "ID_24": "model_zoo/grist_models/ID_24/Mnist_DEBARUS.py",
    "ID_25": "model_zoo/grist_models/ID_25/mnist_DEBARUS.py",
    "ID_26": "model_zoo/grist_models/ID_26/0401_logistic_regression_DEBARUS.py",
    "ID_27": "model_zoo/grist_models/ID_27/0503_softmax_regression_cost_DEBARUS.py",
    "ID_28": "model_zoo/grist_models/ID_28/sc_train_l2reg_div2_DEBARUS.py",
    "ID_29": "model_zoo/grist_models/ID_29/temp_DEBARUS.py",
    "ID_30": "model_zoo/grist_models/ID_30/softmax_DEBARUS.py",
    "ID_31": "model_zoo/grist_models/ID_31/GAN_MNIST_DEBARUS.py",
    "ID_32": "model_zoo/grist_models/ID_32/0504_softmax_regression_DEBARUS.py",
    "ID_33": "model_zoo/grist_models/ID_33/logistic_classification_DEBARUS.py",
    "ID_34": "model_zoo/grist_models/ID_34/softmax_classification_DEBARUS.py",
    "ID_35": "model_zoo/grist_models/ID_35/ch10_04_01_DEBARUS.py",
    "ID_36": "model_zoo/grist_models/ID_36/ch10_04_03_Pic_10_05_DEBARUS.py",
    "ID_37": "model_zoo/grist_models/ID_37/lab-05-1-logistic_regression_DEBARUS.py",
    "ID_38": "model_zoo/grist_models/ID_38/lab-09-1-xor_DEBARUS.py",
    "ID_39": "model_zoo/grist_models/ID_39/ch10_04_04_Pic_10_06_DEBARUS.py",
    "ID_40": "model_zoo/grist_models/ID_40/src/ch5/bmi_DEBARUS.py",
    "ID_41": "model_zoo/grist_models/ID_41/src/ch5/tb-bmi_DEBARUS.py",
    "ID_42": "model_zoo/grist_models/ID_42/src/ch5/tb-bmi2_DEBARUS.py",
    "ID_43": "model_zoo/grist_models/ID_43/ch10_04_05_Pic_10_07_DEBARUS.py",
    "ID_44": "model_zoo/grist_models/ID_44/examples/cnn-mnist/mnist_example_DEBARUS.py",
    "ID_45": "model_zoo/grist_models/ID_45/MNIST_Digit_classification_using_CNN_DEBARUS.py",
    "ID_46": "model_zoo/grist_models/ID_46/torch20_softmax_DEBARUS.py",
    "ID_47": "model_zoo/grist_models/ID_47/torch21_Softmax_Reg1_DEBARUS.py",
    "ID_48": "model_zoo/grist_models/ID_48/day01_DEBARUS.py",
    "ID_49": "model_zoo/grist_models/ID_49/ch10_04_06_Pic_10_08_DEBARUS.py",
    "ID_50": "model_zoo/grist_models/ID_50/DL_1L_DEBARUS.py",
    "ID_51": "model_zoo/grist_models/ID_51/rbm_DEBARUS.py",
    "ID_52": "model_zoo/grist_models/ID_52/5_mnist_full_connection_DEBARUS.py",
    "ID_53": "model_zoo/grist_models/ID_53/h02_20191644_DEBARUS.py",
    "ID_54": "model_zoo/grist_models/ID_54/h03_20191644_DEBARUS.py",
    "ID_55": "model_zoo/grist_models/ID_55/logistic_regression_DEBARUS.py",
    "ID_56": "model_zoo/grist_models/ID_56/01_Logistic_Regression_DEBARUS.py",
    "ID_57": "model_zoo/grist_models/ID_57/03_softmax_cost_DEBARUS.py",
    "ID_58": "model_zoo/grist_models/ID_58/test_DEBARUS.py",
    "ID_59": "model_zoo/grist_models/ID_59/code_06_DEBARUS.py",
    "ID_60": "model_zoo/grist_models/ID_60/simple_saveAndReloadModel_DEBARUS.py",
    "ID_61": "model_zoo/grist_models/ID_61/multi_layer_perceptron_DEBARUS.py",
    "ID_62": "model_zoo/grist_models/ID_62/denoising_RBM_DEBARUS.py",
    "ID_63": "model_zoo/grist_models/ID_63/main_DEBARUS.py"
}

import os
import json
import shutil
import subprocess

def script_scan():
    """
        A one-time function for creating "current_selected_scripts" variable
    :return:
    """
    root_path = 'model_zoo/grist_models'
    dirs = [d for d in os.listdir(root_path) if d.startswith('ID_')]
    dirs = sorted(dirs, key=lambda x: int(x[3:]))

    def _search_script(cur):
        ret = list()
        for thing in os.listdir(cur):
            if os.path.isdir(os.path.join(cur, thing)):
                ret.extend(_search_script(os.path.join(cur, thing)))
            if thing.endswith('grist.py'):
                ret.append(os.path.join(cur, thing))
        return ret

    main_scripts = dict()

    for cur_dir in dirs:
        possible_scripts = _search_script(os.path.join(root_path, cur_dir))
        # print(possible_scripts)
        if len(possible_scripts) == 1:
            main_scripts[cur_dir] = possible_scripts[0]
        elif len(possible_scripts) > 1:
            main_scripts[cur_dir] = possible_scripts[int(input(str(possible_scripts) + '\n'))]
        else:
            raise Exception('weird: there should be at least one GRIST script in each dir')

    # one time use to create "current_selected_scripts" variable
    print(json.dumps(main_scripts, indent=2))


def script_copy():
    current_new_scripts = dict()
    for k, v in current_selected_scripts.items():
        dir_name = os.path.dirname(v)
        file_name = os.path.basename(v)
        kw_lists = ['_grist']
        for kw in kw_lists:
            if file_name.count(kw) > 0:
                file_name = file_name.replace(kw, '')
        assert os.path.exists(dir_name + '/' + file_name)
        new_file_name = file_name.replace('.', '_DEBARUS.')
        print(dir_name + '/' + new_file_name)
        print(os.path.exists(dir_name + '/' + new_file_name))
        if not os.path.exists(dir_name + '/' + new_file_name):
            # copy non-grist scripts
            shutil.copy(dir_name + '/' + file_name, dir_name + '/' + new_file_name)
            print('copy', dir_name + '/' + file_name, 'to', dir_name + '/' + new_file_name)
        current_new_scripts[k] = dir_name + '/' + new_file_name

    # one time use to create "current_new_scripts" variable
    print(json.dumps(current_new_scripts, indent=2))

def script_runner(white_list=[], black_list=[]):
    for id in current_new_scripts:
        num = int(id[3:])
        if len(white_list) > 0 and num not in white_list: continue
        if num in black_list: continue

        # now we can run it
        # TODO


if __name__ == '__main__':
    # script_scan()
    # script_copy()
    script_runner([1])

