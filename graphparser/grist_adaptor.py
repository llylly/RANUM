"""
    The adaptor for GRIST benchmark
    GRIST: https://github.com/Jacob-yen/GRIST

    workflow:
        - update the case_dict to reflect the real directory
        - global change the data path for the benchmark
        - dynamically detect whether it is tf or pytorch model
        - if it is tf model, manually name input/output variables, get the statement after initialize_all_variables, and conduct conversion
        - if it is pytorch model, wrap the model computaiton by nn.Module, then transform to ONNX with pytorch.jit.transform
"""

#
# current_selected_scripts = {
#     "ID_1": "model_zoo/grist_models/ID_1/ghips1_grist.py",
#     "ID_2a": "model_zoo/grist_models/ID_2/command_line/run_logistic_regression_grist.py",
#     "ID_2b": "model_zoo/grist_models/ID_2/command_line/run_logistic_regression_log2_grist.py",
#     "ID_3": "model_zoo/grist_models/ID_3/ghips9_grist.py",
#     "ID_4": "model_zoo/grist_models/ID_4/test/nn/models/test_autoencoder_grist.py",
#     "ID_5": "model_zoo/grist_models/ID_5/tests/test_losses_grist.py",
#     "ID_6": "model_zoo/grist_models/ID_6/soips1_grist.py",
#     "ID_7": "model_zoo/grist_models/ID_7/soips2_grist.py",
#     "ID_8": "model_zoo/grist_models/ID_8/soips6_grist.py",
#     "ID_9a": "model_zoo/grist_models/ID_9/soips7_grist.py",
#     "ID_9b": "model_zoo/grist_models/ID_9/soips7_log2_grist.py",
#     "ID_10": "model_zoo/grist_models/ID_10/soips14_grist.py",
#     "ID_11a": "model_zoo/grist_models/ID_11/nan_model_truediv_grist.py",
#     "ID_11b": "model_zoo/grist_models/ID_11/nan_model_exp_grist.py",
#     "ID_11c": "model_zoo/grist_models/ID_11/nan_model_log_grist.py",
#     "ID_12": "model_zoo/grist_models/ID_12/main_grist.py",
#     "ID_13": "model_zoo/grist_models/ID_13/test/utils/test_softmax_grist.py",
#     "ID_14": "model_zoo/grist_models/ID_14/v3/train_grist.py",
#     "ID_15": "model_zoo/grist_models/ID_15/train_grist.py",
#     "ID_16a": "model_zoo/grist_models/ID_16/nan_model_grist.py",
#     "ID_16b": "model_zoo/grist_models/ID_16/nan_model_log_grist.py",
#     "ID_16c": "model_zoo/grist_models/ID_16/nan_model_truediv_grist.py",
#     "ID_17": "model_zoo/grist_models/ID_17/test_toy_grist.py",
#     "ID_18": "model_zoo/grist_models/ID_18/code_10_image_grist.py",
#     "ID_19": "model_zoo/grist_models/ID_19/mnist_grist.py",
#     "ID_20": "model_zoo/grist_models/ID_20/logistic_regression_grist.py",
#     "ID_21": "model_zoo/grist_models/ID_21/mnist_softmax_grist.py",
#     "ID_22": "model_zoo/grist_models/ID_22/My_pytorch1_grist.py",
#     "ID_23": "model_zoo/grist_models/ID_23/ch5_grist.3_softmax_grist.py",
#     "ID_24": "model_zoo/grist_models/ID_24/Mnist_grist.py",
#     "ID_25": "model_zoo/grist_models/ID_25/mnist_grist.py",
#     "ID_26": "model_zoo/grist_models/ID_26/0401_logistic_regression_grist.py",
#     "ID_27": "model_zoo/grist_models/ID_27/0503_softmax_regression_cost_grist.py",
#     "ID_28a": "model_zoo/grist_models/ID_28/sc_train_l2reg_div2_grist.py",
#     "ID_28b": "model_zoo/grist_models/ID_28/sc_train_l2reg_grist.py",
#     "ID_28c": "model_zoo/grist_models/ID_28/sc_train_creg_grist.py",
#     "ID_28d": "model_zoo/grist_models/ID_28/sc_train_creg_div2_grist.py",
#     "ID_28e": "model_zoo/grist_models/ID_28/sc_train_creg_div4_grist.py",
#     "ID_29": "model_zoo/grist_models/ID_29/temp_grist.py",
#     "ID_30": "model_zoo/grist_models/ID_30/softmax_grist.py",
#     "ID_31": "model_zoo/grist_models/ID_31/GAN_MNIST_grist.py",
#     "ID_32": "model_zoo/grist_models/ID_32/0504_softmax_regression_grist.py",
#     "ID_33": "model_zoo/grist_models/ID_33/logistic_classification_grist.py",
#     "ID_34": "model_zoo/grist_models/ID_34/softmax_classification_grist.py",
#     "ID_35a": "model_zoo/grist_models/ID_35/ch10_04_01_exp_grist.py",
#     "ID_35b": "model_zoo/grist_models/ID_35/ch10_04_01_sqrt_grist.py",
#     "ID_35c": "model_zoo/grist_models/ID_35/ch10_04_01_grist.py",
#     "ID_36a": "model_zoo/grist_models/ID_36/ch10_04_03_Pic_10_05_grist.py",
#     "ID_36b": "model_zoo/grist_models/ID_36/ch10_04_03_Pic_10_05_exp_grist.py",
#     "ID_36c": "model_zoo/grist_models/ID_36/ch10_04_03_Pic_10_05_sqrt_grist.py",
#     "ID_37": "model_zoo/grist_models/ID_37/lab-05-1-logistic_regression_grist.py",
#     "ID_38": "model_zoo/grist_models/ID_38/lab-09-1-xor_grist.py",
#     "ID_39a": "model_zoo/grist_models/ID_39/ch10_04_04_Pic_10_06_grist.py",
#     "ID_39b": "model_zoo/grist_models/ID_39/ch10_04_04_Pic_10_06_exp_grist.py",
#     "ID_39c": "model_zoo/grist_models/ID_39/ch10_04_04_Pic_10_06_sqrt_grist.py",
#     "ID_40": "model_zoo/grist_models/ID_40/src/ch5/bmi_grist.py",
#     "ID_41": "model_zoo/grist_models/ID_41/src/ch5/tb-bmi_grist.py",
#     "ID_42": "model_zoo/grist_models/ID_42/src/ch5/tb-bmi2_grist.py",
#     "ID_43a": "model_zoo/grist_models/ID_43/ch10_04_05_Pic_10_07_grist.py",
#     "ID_43b": "model_zoo/grist_models/ID_43/ch10_04_05_Pic_10_07_exp_grist.py",
#     "ID_43c": "model_zoo/grist_models/ID_43/ch10_04_05_Pic_10_07_sqrt_grist.py",
#     "ID_44": "model_zoo/grist_models/ID_44/examples/cnn-mnist/mnist_example_grist.py",
#     "ID_45a": "model_zoo/grist_models/ID_45/MNIST_Digit_classification_using_CNN_grist.py",
#     "ID_45b": "model_zoo/grist_models/ID_45/MNIST_Digit_classification_using_CNN2_grist.py",
#     "ID_46": "model_zoo/grist_models/ID_46/torch20_softmax_grist.py",
#     "ID_47": "model_zoo/grist_models/ID_47/torch21_Softmax_Reg1_grist.py",
#     "ID_48a": "model_zoo/grist_models/ID_48/day01_grist.py",
#     "ID_48b": "model_zoo/grist_models/ID_48/day01_2_grist.py",
#     "ID_49a": "model_zoo/grist_models/ID_49/ch10_04_06_Pic_10_08_grist.py",
#     "ID_49b": "model_zoo/grist_models/ID_49/ch10_04_06_Pic_10_08_exp_grist.py",
#     "ID_49c": "model_zoo/grist_models/ID_49/ch10_04_06_Pic_10_08_sqrt_grist.py",
#     "ID_50": "model_zoo/grist_models/ID_50/DL_1L_grist.py",
#     "ID_51": "model_zoo/grist_models/ID_51/rbm_grist.py",
#     "ID_52": "model_zoo/grist_models/ID_52/5_mnist_full_connection_grist.py",
#     "ID_53": "model_zoo/grist_models/ID_53/h02_20191644_grist.py",
#     "ID_54": "model_zoo/grist_models/ID_54/h03_20191644_grist.py",
#     "ID_55": "model_zoo/grist_models/ID_55/logistic_regression_grist.py",
#     "ID_56": "model_zoo/grist_models/ID_56/01_Logistic_Regression_grist.py",
#     "ID_57": "model_zoo/grist_models/ID_57/03_softmax_cost_grist.py",
#     "ID_58": "model_zoo/grist_models/ID_58/test_grist.py",
#     "ID_59": "model_zoo/grist_models/ID_59/code_06_grist.py",
#     "ID_60": "model_zoo/grist_models/ID_60/simple_saveAndReloadModel_grist.py",
#     "ID_61": "model_zoo/grist_models/ID_61/multi_layer_perceptron_grist.py",
#     "ID_62": "model_zoo/grist_models/ID_62/denoising_RBM_grist.py",
#     "ID_63": "model_zoo/grist_models/ID_63/main_grist.py"
# }
#
# current_new_scripts = {
#     "ID_1": "model_zoo/grist_models/ID_1/ghips1_RANUM.py",
#     "ID_2a": "model_zoo/grist_models/ID_2/command_line/run_logistic_regression_RANUM.py",
#     "ID_2b": "model_zoo/grist_models/ID_2/command_line/run_logistic_regression_log2_RANUM.py",
#     "ID_3": "model_zoo/grist_models/ID_3/ghips9_RANUM.py",
#     "ID_4": "model_zoo/grist_models/ID_4/test/nn/models/test_autoencoder_RANUM.py",
#     "ID_5": "model_zoo/grist_models/ID_5/tests/test_losses_RANUM.py",
#     "ID_6": "model_zoo/grist_models/ID_6/soips1_RANUM.py",
#     "ID_7": "model_zoo/grist_models/ID_7/soips2_RANUM.py",
#     "ID_8": "model_zoo/grist_models/ID_8/soips6_RANUM.py",
#     "ID_9a": "model_zoo/grist_models/ID_9/soips7_RANUM.py",
#     "ID_9b": "model_zoo/grist_models/ID_9/soips7_log2_RANUM.py",
#     "ID_10": "model_zoo/grist_models/ID_10/soips14_RANUM.py",
#     "ID_11a": "model_zoo/grist_models/ID_11/nan_model_truediv_RANUM.py",
#     "ID_11b": "model_zoo/grist_models/ID_11/nan_model_exp_RANUM.py",
#     "ID_11c": "model_zoo/grist_models/ID_11/nan_model_log_RANUM.py",
#     "ID_12": "model_zoo/grist_models/ID_12/main_RANUM.py",
#     "ID_13": "model_zoo/grist_models/ID_13/test/utils/test_softmax_RANUM.py",
#     "ID_14": "model_zoo/grist_models/ID_14/v3/train_RANUM.py",
#     "ID_15": "model_zoo/grist_models/ID_15/train_RANUM.py",
#     "ID_16a": "model_zoo/grist_models/ID_16/nan_model_RANUM.py",
#     "ID_16b": "model_zoo/grist_models/ID_16/nan_model_log_RANUM.py",
#     "ID_16c": "model_zoo/grist_models/ID_16/nan_model_truediv_RANUM.py",
#     "ID_17": "model_zoo/grist_models/ID_17/test_toy_RANUM.py",
#     "ID_18": "model_zoo/grist_models/ID_18/code_10_image_RANUM.py",
#     "ID_19": "model_zoo/grist_models/ID_19/mnist_RANUM.py",
#     "ID_20": "model_zoo/grist_models/ID_20/logistic_regression_RANUM.py",
#     "ID_21": "model_zoo/grist_models/ID_21/mnist_softmax_RANUM.py",
#     "ID_22": "model_zoo/grist_models/ID_22/My_pytorch1_RANUM.py",
#     "ID_23": "model_zoo/grist_models/ID_23/ch5_RANUM.3_softmax_RANUM.py",
#     "ID_24": "model_zoo/grist_models/ID_24/Mnist_RANUM.py",
#     "ID_25": "model_zoo/grist_models/ID_25/mnist_RANUM.py",
#     "ID_26": "model_zoo/grist_models/ID_26/0401_logistic_regression_RANUM.py",
#     "ID_27": "model_zoo/grist_models/ID_27/0503_softmax_regression_cost_RANUM.py",
#     "ID_28a": "model_zoo/grist_models/ID_28/sc_train_l2reg_div2_RANUM.py",
#     "ID_28b": "model_zoo/grist_models/ID_28/sc_train_l2reg_RANUM.py",
#     "ID_28c": "model_zoo/grist_models/ID_28/sc_train_creg_RANUM.py",
#     "ID_28d": "model_zoo/grist_models/ID_28/sc_train_creg_div2_RANUM.py",
#     "ID_28e": "model_zoo/grist_models/ID_28/sc_train_creg_div4_RANUM.py",
#     "ID_29": "model_zoo/grist_models/ID_29/temp_RANUM.py",
#     "ID_30": "model_zoo/grist_models/ID_30/softmax_RANUM.py",
#     "ID_31": "model_zoo/grist_models/ID_31/GAN_MNIST_RANUM.py",
#     "ID_32": "model_zoo/grist_models/ID_32/0504_softmax_regression_RANUM.py",
#     "ID_33": "model_zoo/grist_models/ID_33/logistic_classification_RANUM.py",
#     "ID_34": "model_zoo/grist_models/ID_34/softmax_classification_RANUM.py",
#     "ID_35a": "model_zoo/grist_models/ID_35/ch10_04_01_exp_RANUM.py",
#     "ID_35b": "model_zoo/grist_models/ID_35/ch10_04_01_sqrt_RANUM.py",
#     "ID_35c": "model_zoo/grist_models/ID_35/ch10_04_01_RANUM.py",
#     "ID_36a": "model_zoo/grist_models/ID_36/ch10_04_03_Pic_10_05_RANUM.py",
#     "ID_36b": "model_zoo/grist_models/ID_36/ch10_04_03_Pic_10_05_exp_RANUM.py",
#     "ID_36c": "model_zoo/grist_models/ID_36/ch10_04_03_Pic_10_05_sqrt_RANUM.py",
#     "ID_37": "model_zoo/grist_models/ID_37/lab-05-1-logistic_regression_RANUM.py",
#     "ID_38": "model_zoo/grist_models/ID_38/lab-09-1-xor_RANUM.py",
#     "ID_39a": "model_zoo/grist_models/ID_39/ch10_04_04_Pic_10_06_RANUM.py",
#     "ID_39b": "model_zoo/grist_models/ID_39/ch10_04_04_Pic_10_06_exp_RANUM.py",
#     "ID_39c": "model_zoo/grist_models/ID_39/ch10_04_04_Pic_10_06_sqrt_RANUM.py",
#     "ID_40": "model_zoo/grist_models/ID_40/src/ch5/bmi_RANUM.py",
#     "ID_41": "model_zoo/grist_models/ID_41/src/ch5/tb-bmi_RANUM.py",
#     "ID_42": "model_zoo/grist_models/ID_42/src/ch5/tb-bmi2_RANUM.py",
#     "ID_43a": "model_zoo/grist_models/ID_43/ch10_04_05_Pic_10_07_RANUM.py",
#     "ID_43b": "model_zoo/grist_models/ID_43/ch10_04_05_Pic_10_07_exp_RANUM.py",
#     "ID_43c": "model_zoo/grist_models/ID_43/ch10_04_05_Pic_10_07_sqrt_RANUM.py",
#     "ID_44": "model_zoo/grist_models/ID_44/examples/cnn-mnist/mnist_example_RANUM.py",
#     "ID_45a": "model_zoo/grist_models/ID_45/MNIST_Digit_classification_using_CNN_RANUM.py",
#     "ID_45b": "model_zoo/grist_models/ID_45/MNIST_Digit_classification_using_CNN2_RANUM.py",
#     "ID_46": "model_zoo/grist_models/ID_46/torch20_softmax_RANUM.py",
#     "ID_47": "model_zoo/grist_models/ID_47/torch21_Softmax_Reg1_RANUM.py",
#     "ID_48a": "model_zoo/grist_models/ID_48/day01_RANUM.py",
#     "ID_48b": "model_zoo/grist_models/ID_48/day01_2_RANUM.py",
#     "ID_49a": "model_zoo/grist_models/ID_49/ch10_04_06_Pic_10_08_RANUM.py",
#     "ID_49b": "model_zoo/grist_models/ID_49/ch10_04_06_Pic_10_08_exp_RANUM.py",
#     "ID_49c": "model_zoo/grist_models/ID_49/ch10_04_06_Pic_10_08_sqrt_RANUM.py",
#     "ID_50": "model_zoo/grist_models/ID_50/DL_1L_RANUM.py",
#     "ID_51": "model_zoo/grist_models/ID_51/rbm_RANUM.py",
#     "ID_52": "model_zoo/grist_models/ID_52/5_mnist_full_connection_RANUM.py",
#     "ID_53": "model_zoo/grist_models/ID_53/h02_20191644_RANUM.py",
#     "ID_54": "model_zoo/grist_models/ID_54/h03_20191644_RANUM.py",
#     "ID_55": "model_zoo/grist_models/ID_55/logistic_regression_RANUM.py",
#     "ID_56": "model_zoo/grist_models/ID_56/01_Logistic_Regression_RANUM.py",
#     "ID_57": "model_zoo/grist_models/ID_57/03_softmax_cost_RANUM.py",
#     "ID_58": "model_zoo/grist_models/ID_58/test_RANUM.py",
#     "ID_59": "model_zoo/grist_models/ID_59/code_06_RANUM.py",
#     "ID_60": "model_zoo/grist_models/ID_60/simple_saveAndReloadModel_RANUM.py",
#     "ID_61": "model_zoo/grist_models/ID_61/multi_layer_perceptron_RANUM.py",
#     "ID_62": "model_zoo/grist_models/ID_62/denoising_RBM_RANUM.py",
#     "ID_63": "model_zoo/grist_models/ID_63/main_RANUM.py"
# }
#
# package_info = {
#     "ID_1": "tensorflow",
#     "ID_2a": "tensorflow",
#     "ID_2b": "tensorflow",
#     "ID_3": "tensorflow",
#     "ID_4": "torch",
#     "ID_5": "torch",
#     "ID_6": "tensorflow",
#     "ID_7": "tensorflow",
#     "ID_8": "tensorflow",
#     "ID_9a": "tensorflow",
#     "ID_9b": "tensorflow",
#     "ID_10": "tensorflow",
#     "ID_11a": "tensorflow",
#     "ID_11b": "tensorflow",
#     "ID_11c": "tensorflow",
#     "ID_12": "torch",
#     "ID_13": "torch",
#     "ID_14": "tensorflow",
#     "ID_15": "tensorflow",
#     "ID_16a": "tensorflow",
#     "ID_16b": "tensorflow",
#     "ID_16c": "tensorflow",
#     "ID_17": "tensorflow",
#     "ID_18": "tensorflow",
#     "ID_19": "tensorflow",
#     "ID_20": "tensorflow",
#     "ID_21": "tensorflow",
#     "ID_22": "torch",
#     "ID_23": "torch",
#     "ID_24": "tensorflow",
#     "ID_25": "tensorflow",
#     "ID_26": "torch",
#     "ID_27": "torch",
#     "ID_28a": "tensorflow",
#     "ID_28b": "tensorflow",
#     "ID_28c": "tensorflow",
#     "ID_28d": "tensorflow",
#     "ID_28e": "tensorflow",
#     "ID_29": "tensorflow",
#     "ID_30": "tensorflow",
#     "ID_31": "tensorflow",
#     "ID_32": "torch",
#     "ID_33": "torch",
#     "ID_34": "torch",
#     "ID_35a": "tensorflow",
#     "ID_35b": "tensorflow",
#     "ID_35c": "tensorflow",
#     "ID_36a": "tensorflow",
#     "ID_36b": "tensorflow",
#     "ID_36c": "tensorflow",
#     "ID_37": "torch",
#     "ID_38": "torch",
#     "ID_39a": "tensorflow",
#     "ID_39b": "tensorflow",
#     "ID_39c": "tensorflow",
#     "ID_40": "tensorflow",
#     "ID_41": "tensorflow",
#     "ID_42": "tensorflow",
#     "ID_43a": "tensorflow",
#     "ID_43b": "tensorflow",
#     "ID_43c": "tensorflow",
#     "ID_44": "tensorflow",
#     "ID_45a": "tensorflow",
#     "ID_45b": "tensorflow",
#     "ID_46": "torch",
#     "ID_47": "torch",
#     "ID_48a": "tensorflow",
#     "ID_48b": "tensorflow",
#     "ID_49a": "tensorflow",
#     "ID_49b": "tensorflow",
#     "ID_49c": "tensorflow",
#     "ID_50": "tensorflow",
#     "ID_51": "torch",
#     "ID_52": "tensorflow",
#     "ID_53": "torch",
#     "ID_54": "torch",
#     "ID_55": "tensorflow",
#     "ID_56": "torch",
#     "ID_57": "torch",
#     "ID_58": "tensorflow",
#     "ID_59": "tensorflow",
#     "ID_60": "tensorflow",
#     "ID_61": "tensorflow",
#     "ID_62": "torch",
#     "ID_63": "torch"
# }
#
# run_ids = ['63']
#
# # only for tensorflow models
# inputs_outputs = {
#     "ID_1": (['x', 'y', 'keep_prob'], ['cross_entropy', 'obj_function']),
#     "ID_2a": (['x-input', 'y-input'], ['cost/cost', 'cost/obj_function']),
#     "ID_2b": (['x-input', 'y-input'], ['cost/cost', 'cost/obj_function']),
#     "ID_3": (['x', 'y'], ['cost', 'obj_function']),
#     "ID_6": (['x', 'y', 'keep_prob'], ['cross_entropy']),
#     "ID_7": (['x', 'y', 'keep_prob'], ['cross_entropy', 'obj_function']),
#     "ID_8": (['x', 'y', 'keep_prob'], ['cross_entropy', 'obj_function']),
#     "ID_9a": (['x', 'y'], ['loss', 'obj_function']),
#     "ID_9b": (['x', 'y'], ['loss', 'obj_function']),
#     "ID_10": (['x', 'y'], ['cross_entropy', 'obj_function']),
#     # seems the DynamicStitch is still not supported by tf2onnx
#     # "ID_11": (['x', 'y'], ['loss', 'obj_function', 'obj_grad']),
#     "ID_11a": (['x', 'y'], ['loss', 'obj_function']),
#     "ID_11b": (['x', 'y'], ['loss', 'obj_function']),
#     "ID_11c": (['x', 'y'], ['loss', 'obj_function']),
#     "ID_14": (['input_x', 'label'], ['loss']),
#     "ID_15": (['X', 'S1', 'S2', 'y'], ['cost', 'err', 'obj_function']),
#     "ID_16a": (['x', 'y'], ['loss', 'obj_function']),
#     "ID_16b": (['x', 'y'], ['loss', 'obj_function']),
#     "ID_16c": (['x', 'y'], ['loss', 'obj_function']),
#     "ID_17": (['x', 'y'], ['cross_entropy', 'loss', 'accuracy', 'obj_function']),
#     "ID_18": (['x', 'labels'], ['loss/cross_entropy', 'obj_function']),
#     "ID_19": (['x_input', 'y'], ['cross_entropy', 'obj_var']),
#     "ID_20": (['x', 'y'], ['cost', 'obj_var']),
#     "ID_21": (['x', 'y'], ['cross_entropy', 'obj_function']),
#     "ID_24": (['input/x-input', 'input/y-input'], ['cross_entropy', 'obj_function']),
#     "ID_25": (['x', 'y'], ['cross_entropy', 'obj_var']),
#     "ID_28a": (['features', 'labels'], ['loss']),
#     "ID_28b": (['features', 'labels'], ['loss', 'obj_function']),
#     "ID_28c": (['features', 'labels'], ['loss', 'obj_function']),
#     "ID_28d": (['features', 'labels'], ['loss', 'obj_function']),
#     "ID_28e": (['features', 'labels'], ['loss', 'obj_function']),
#     "ID_29": (['x', 'y_'], ['cross_entropy', 'obj_function']),
#     "ID_30": (['x', 'y_'], ['cross_entropy', 'obj_function', 'obj_var']),
#     "ID_31": (['x', 'z', 'keep_prob'], ['D_loss', 'obj_function']),
#     "ID_35a": (['x', 'eps'], ['cost', 'obj_function']),
#     "ID_35b": (['x', 'eps'], ['cost', 'obj_function']),
#     "ID_35c": (['x', 'eps'], ['cost', 'obj_function']),
#     "ID_36a": (['x', 'eps'], ['cost', 'obj_function']),
#     "ID_36b": (['x', 'eps'], ['cost', 'obj_function']),
#     "ID_36c": (['x', 'eps'], ['cost', 'obj_function']),
#     "ID_39a": (['x', 'eps'], ['cost', 'obj_function']),
#     "ID_39b": (['x', 'eps'], ['cost', 'obj_function']),
#     "ID_39c": (['x', 'eps'], ['cost', 'obj_function']),
#     "ID_40": (['x', 'y_'], ['cross_entropy']),
#     "ID_41": (['x', 'y_'], ['cross_entropy', 'obj_function']),
#     "ID_42": (['x', 'y_'], ['loss/cross_entropy', 'obj_function']),
#     "ID_43a": (['x', 'y', 'eps'], ['cost', 'obj_function']),
#     "ID_43b": (['x', 'y', 'eps'], ['cost', 'obj_function']),
#     "ID_43c": (['x', 'y', 'eps'], ['cost', 'obj_function']),
#     "ID_44": (['x', 'y_'], ['cross_entropy', 'obj_function']),
#     "ID_45a": (['x', 'y_'], ['cross_entropy', 'obj_function']),
#     "ID_45b": (['x', 'y_'], ['cross_entropy', 'obj_function']),
#     "ID_48a": (['x', 'y_true', 'keep_prob'], ['cross_entropy', 'obj_function']),
#     "ID_48b": (['x', 'y_true'], ['cross_entropy', 'obj_function']),
#     "ID_49a": (['x', 'y', 'eps'], ['cost', 'obj_function']),
#     "ID_49b": (['x', 'y', 'eps'], ['cost', 'obj_function']),
#     "ID_49c": (['x', 'y', 'eps'], ['cost', 'obj_function']),
#     "ID_50": (['x', 'y_'], ['cross_entropy', 'obj_function']),
#     "ID_52": (['x', 'y_'], ['cross_entropy', 'obj_function']),
#     "ID_55": (['x', 'y', 'keep_prob'], ['cross_entropy', 'obj_function']),
#     "ID_58": (['x', 'y_', 'keep_prob'], ['cross_entropy', 'obj_function']),
#     "ID_59": (['x', 'y'], ['cross_entropy', 'obj_function']),
#     "ID_60": (['x', 'y_'], ['cross_entropy', 'obj_function']),
#     "ID_61": (['x', 'y_', 'keep_prob'], ['cross_entropy', 'obj_function'])
# }

# generate new models that match GRIST paper exactly

current_selected_scripts = {
    "ID_1": "model_zoo/grist_models/ID_1/ghips1_grist.py",
    "ID_2a": "model_zoo/grist_models/ID_2/command_line/run_logistic_regression_grist.py",
    "ID_2b": "model_zoo/grist_models/ID_2/command_line/run_logistic_regression_log2_grist.py",
    "ID_3": "model_zoo/grist_models/ID_3/ghips9_grist.py",
    "ID_4": "model_zoo/grist_models/ID_4/test/nn/models/test_autoencoder_grist.py",
    "ID_5": "model_zoo/grist_models/ID_5/tests/test_losses_grist.py",
    "ID_6": "model_zoo/grist_models/ID_6/soips1_grist.py",
    "ID_7": "model_zoo/grist_models/ID_7/soips2_grist.py",
    "ID_8": "model_zoo/grist_models/ID_8/soips6_grist.py",
    "ID_9a": "model_zoo/grist_models/ID_9/soips7_grist.py",
    "ID_9b": "model_zoo/grist_models/ID_9/soips7_log2_grist.py",
    "ID_10": "model_zoo/grist_models/ID_10/soips14_grist.py",
    "ID_11a": "model_zoo/grist_models/ID_11/nan_model_truediv_grist.py",
    "ID_11b": "model_zoo/grist_models/ID_11/nan_model_exp_grist.py",
    "ID_11c": "model_zoo/grist_models/ID_11/nan_model_log_grist.py",
    "ID_12": "model_zoo/grist_models/ID_12/main_grist.py",
    "ID_13": "model_zoo/grist_models/ID_13/test/utils/test_softmax_grist.py",
    "ID_14": "model_zoo/grist_models/ID_14/v3/train_grist.py",
    "ID_15": "model_zoo/grist_models/ID_15/train_grist.py",
    "ID_16a": "model_zoo/grist_models/ID_16/nan_model_grist.py",
    "ID_16b": "model_zoo/grist_models/ID_16/nan_model_log_grist.py",
    "ID_16c": "model_zoo/grist_models/ID_16/nan_model_truediv_grist.py",
    "ID_17": "model_zoo/grist_models/ID_17/test_toy_grist.py",
    "ID_18": "model_zoo/grist_models/ID_18/code_10_image_grist.py",
    "ID_19": "model_zoo/grist_models/ID_19/mnist_grist.py",
    "ID_20": "model_zoo/grist_models/ID_20/logistic_regression_grist.py",
    "ID_21": "model_zoo/grist_models/ID_21/mnist_softmax_grist.py",
    "ID_22": "model_zoo/grist_models/ID_22/My_pytorch1_grist.py",
    "ID_23": "model_zoo/grist_models/ID_23/ch5_grist.3_softmax_grist.py",
    "ID_24": "model_zoo/grist_models/ID_24/Mnist_grist.py",
    "ID_25": "model_zoo/grist_models/ID_25/mnist_grist.py",
    "ID_26": "model_zoo/grist_models/ID_26/0401_logistic_regression_grist.py",
    "ID_27": "model_zoo/grist_models/ID_27/0503_softmax_regression_cost_grist.py",
    "ID_28d": "model_zoo/grist_models/ID_28/sc_train_l2reg_div2_grist.py",
    "ID_28c": "model_zoo/grist_models/ID_28/sc_train_l2reg_grist.py",
    "ID_28a": "model_zoo/grist_models/ID_28/sc_train_creg_grist.py",
    "ID_28b": "model_zoo/grist_models/ID_28/sc_train_creg_div2_grist.py",
    # "ID_28e": "model_zoo/grist_models/ID_28/sc_train_creg_div4_grist.py",
    "ID_29": "model_zoo/grist_models/ID_29/temp_grist.py",
    "ID_30": "model_zoo/grist_models/ID_30/softmax_grist.py",
    "ID_31": "model_zoo/grist_models/ID_31/GAN_MNIST_grist.py",
    "ID_32": "model_zoo/grist_models/ID_32/0504_softmax_regression_grist.py",
    "ID_33": "model_zoo/grist_models/ID_33/logistic_classification_grist.py",
    "ID_34": "model_zoo/grist_models/ID_34/softmax_classification_grist.py",
    # "ID_35a": "model_zoo/grist_models/ID_35/ch10_04_01_exp_grist.py",
    "ID_35b": "model_zoo/grist_models/ID_35/ch10_04_01_sqrt_grist.py",
    "ID_35a": "model_zoo/grist_models/ID_35/ch10_04_01_grist.py",
    "ID_36a": "model_zoo/grist_models/ID_36/ch10_04_03_Pic_10_05_grist.py",
    # "ID_36b": "model_zoo/grist_models/ID_36/ch10_04_03_Pic_10_05_exp_grist.py",
    "ID_36b": "model_zoo/grist_models/ID_36/ch10_04_03_Pic_10_05_sqrt_grist.py",
    "ID_37": "model_zoo/grist_models/ID_37/lab-05-1-logistic_regression_grist.py",
    "ID_38": "model_zoo/grist_models/ID_38/lab-09-1-xor_grist.py",
    "ID_39a": "model_zoo/grist_models/ID_39/ch10_04_04_Pic_10_06_grist.py",
    # "ID_39b": "model_zoo/grist_models/ID_39/ch10_04_04_Pic_10_06_exp_grist.py",
    "ID_39b": "model_zoo/grist_models/ID_39/ch10_04_04_Pic_10_06_sqrt_grist.py",
    "ID_40": "model_zoo/grist_models/ID_40/src/ch5/bmi_grist.py",
    "ID_41": "model_zoo/grist_models/ID_41/src/ch5/tb-bmi_grist.py",
    "ID_42": "model_zoo/grist_models/ID_42/src/ch5/tb-bmi2_grist.py",
    "ID_43a": "model_zoo/grist_models/ID_43/ch10_04_05_Pic_10_07_grist.py",
    # "ID_43b": "model_zoo/grist_models/ID_43/ch10_04_05_Pic_10_07_exp_grist.py",
    "ID_43b": "model_zoo/grist_models/ID_43/ch10_04_05_Pic_10_07_sqrt_grist.py",
    "ID_44": "model_zoo/grist_models/ID_44/examples/cnn-mnist/mnist_example_grist.py",
    "ID_45a": "model_zoo/grist_models/ID_45/MNIST_Digit_classification_using_CNN_grist.py",
    "ID_45b": "model_zoo/grist_models/ID_45/MNIST_Digit_classification_using_CNN2_grist.py",
    "ID_46": "model_zoo/grist_models/ID_46/torch20_softmax_grist.py",
    "ID_47": "model_zoo/grist_models/ID_47/torch21_Softmax_Reg1_grist.py",
    "ID_48a": "model_zoo/grist_models/ID_48/day01_grist.py",
    "ID_48b": "model_zoo/grist_models/ID_48/day01_2_grist.py",
    "ID_49a": "model_zoo/grist_models/ID_49/ch10_04_06_Pic_10_08_grist.py",
    # "ID_49b": "model_zoo/grist_models/ID_49/ch10_04_06_Pic_10_08_exp_grist.py",
    "ID_49b": "model_zoo/grist_models/ID_49/ch10_04_06_Pic_10_08_sqrt_grist.py",
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
    "ID_1": "model_zoo/grist_models/ID_1/ghips1_RANUM.py",
    "ID_2a": "model_zoo/grist_models/ID_2/command_line/run_logistic_regression_RANUM.py",
    "ID_2b": "model_zoo/grist_models/ID_2/command_line/run_logistic_regression_log2_RANUM.py",
    "ID_3": "model_zoo/grist_models/ID_3/ghips9_RANUM.py",
    "ID_4": "model_zoo/grist_models/ID_4/test/nn/models/test_autoencoder_RANUM.py",
    "ID_5": "model_zoo/grist_models/ID_5/tests/test_losses_RANUM.py",
    "ID_6": "model_zoo/grist_models/ID_6/soips1_RANUM.py",
    "ID_7": "model_zoo/grist_models/ID_7/soips2_RANUM.py",
    "ID_8": "model_zoo/grist_models/ID_8/soips6_RANUM.py",
    "ID_9a": "model_zoo/grist_models/ID_9/soips7_RANUM.py",
    "ID_9b": "model_zoo/grist_models/ID_9/soips7_log2_RANUM.py",
    "ID_10": "model_zoo/grist_models/ID_10/soips14_RANUM.py",
    "ID_11a": "model_zoo/grist_models/ID_11/nan_model_truediv_RANUM.py",
    "ID_11b": "model_zoo/grist_models/ID_11/nan_model_exp_RANUM.py",
    "ID_11c": "model_zoo/grist_models/ID_11/nan_model_log_RANUM.py",
    "ID_12": "model_zoo/grist_models/ID_12/main_RANUM.py",
    "ID_13": "model_zoo/grist_models/ID_13/test/utils/test_softmax_RANUM.py",
    "ID_14": "model_zoo/grist_models/ID_14/v3/train_RANUM.py",
    "ID_15": "model_zoo/grist_models/ID_15/train_RANUM.py",
    "ID_16a": "model_zoo/grist_models/ID_16/nan_model_RANUM.py",
    "ID_16b": "model_zoo/grist_models/ID_16/nan_model_log_RANUM.py",
    "ID_16c": "model_zoo/grist_models/ID_16/nan_model_truediv_RANUM.py",
    "ID_17": "model_zoo/grist_models/ID_17/test_toy_RANUM.py",
    "ID_18": "model_zoo/grist_models/ID_18/code_10_image_RANUM.py",
    "ID_19": "model_zoo/grist_models/ID_19/mnist_RANUM.py",
    "ID_20": "model_zoo/grist_models/ID_20/logistic_regression_RANUM.py",
    "ID_21": "model_zoo/grist_models/ID_21/mnist_softmax_RANUM.py",
    "ID_22": "model_zoo/grist_models/ID_22/My_pytorch1_RANUM.py",
    "ID_23": "model_zoo/grist_models/ID_23/ch5_RANUM.3_softmax_RANUM.py",
    "ID_24": "model_zoo/grist_models/ID_24/Mnist_RANUM.py",
    "ID_25": "model_zoo/grist_models/ID_25/mnist_RANUM.py",
    "ID_26": "model_zoo/grist_models/ID_26/0401_logistic_regression_RANUM.py",
    "ID_27": "model_zoo/grist_models/ID_27/0503_softmax_regression_cost_RANUM.py",
    "ID_28d": "model_zoo/grist_models/ID_28/sc_train_l2reg_div2_RANUM.py",
    "ID_28c": "model_zoo/grist_models/ID_28/sc_train_l2reg_RANUM.py",
    "ID_28a": "model_zoo/grist_models/ID_28/sc_train_creg_RANUM.py",
    "ID_28b": "model_zoo/grist_models/ID_28/sc_train_creg_div2_RANUM.py",
    # "ID_28e": "model_zoo/grist_models/ID_28/sc_train_creg_div4_RANUM.py",
    "ID_29": "model_zoo/grist_models/ID_29/temp_RANUM.py",
    "ID_30": "model_zoo/grist_models/ID_30/softmax_RANUM.py",
    "ID_31": "model_zoo/grist_models/ID_31/GAN_MNIST_RANUM.py",
    "ID_32": "model_zoo/grist_models/ID_32/0504_softmax_regression_RANUM.py",
    "ID_33": "model_zoo/grist_models/ID_33/logistic_classification_RANUM.py",
    "ID_34": "model_zoo/grist_models/ID_34/softmax_classification_RANUM.py",
    # "ID_35a": "model_zoo/grist_models/ID_35/ch10_04_01_exp_RANUM.py",
    "ID_35b": "model_zoo/grist_models/ID_35/ch10_04_01_sqrt_RANUM.py",
    "ID_35a": "model_zoo/grist_models/ID_35/ch10_04_01_RANUM.py",
    "ID_36a": "model_zoo/grist_models/ID_36/ch10_04_03_Pic_10_05_RANUM.py",
    # "ID_36b": "model_zoo/grist_models/ID_36/ch10_04_03_Pic_10_05_exp_RANUM.py",
    "ID_36b": "model_zoo/grist_models/ID_36/ch10_04_03_Pic_10_05_sqrt_RANUM.py",
    "ID_37": "model_zoo/grist_models/ID_37/lab-05-1-logistic_regression_RANUM.py",
    "ID_38": "model_zoo/grist_models/ID_38/lab-09-1-xor_RANUM.py",
    "ID_37z": "model_zoo/grist_models/ID_37/lab-05-1-logistic_regression_RANUM_fix.py",
    "ID_38z": "model_zoo/grist_models/ID_38/lab-09-1-xor_RANUM_fix.py",
    "ID_39a": "model_zoo/grist_models/ID_39/ch10_04_04_Pic_10_06_RANUM.py",
    # "ID_39b": "model_zoo/grist_models/ID_39/ch10_04_04_Pic_10_06_exp_RANUM.py",
    "ID_39b": "model_zoo/grist_models/ID_39/ch10_04_04_Pic_10_06_sqrt_RANUM.py",
    "ID_40": "model_zoo/grist_models/ID_40/src/ch5/bmi_RANUM.py",
    "ID_41": "model_zoo/grist_models/ID_41/src/ch5/tb-bmi_RANUM.py",
    "ID_42": "model_zoo/grist_models/ID_42/src/ch5/tb-bmi2_RANUM.py",
    "ID_43a": "model_zoo/grist_models/ID_43/ch10_04_05_Pic_10_07_RANUM.py",
    # "ID_43b": "model_zoo/grist_models/ID_43/ch10_04_05_Pic_10_07_exp_RANUM.py",
    "ID_43b": "model_zoo/grist_models/ID_43/ch10_04_05_Pic_10_07_sqrt_RANUM.py",
    "ID_44": "model_zoo/grist_models/ID_44/examples/cnn-mnist/mnist_example_RANUM.py",
    "ID_45a": "model_zoo/grist_models/ID_45/MNIST_Digit_classification_using_CNN_RANUM.py",
    "ID_45b": "model_zoo/grist_models/ID_45/MNIST_Digit_classification_using_CNN2_RANUM.py",
    "ID_46": "model_zoo/grist_models/ID_46/torch20_softmax_RANUM.py",
    "ID_47": "model_zoo/grist_models/ID_47/torch21_Softmax_Reg1_RANUM.py",
    "ID_48a": "model_zoo/grist_models/ID_48/day01_RANUM.py",
    "ID_48b": "model_zoo/grist_models/ID_48/day01_2_RANUM.py",
    "ID_49a": "model_zoo/grist_models/ID_49/ch10_04_06_Pic_10_08_RANUM.py",
    # "ID_49b": "model_zoo/grist_models/ID_49/ch10_04_06_Pic_10_08_exp_RANUM.py",
    "ID_49b": "model_zoo/grist_models/ID_49/ch10_04_06_Pic_10_08_sqrt_RANUM.py",
    "ID_50": "model_zoo/grist_models/ID_50/DL_1L_RANUM.py",
    "ID_51": "model_zoo/grist_models/ID_51/rbm_RANUM.py",
    "ID_52": "model_zoo/grist_models/ID_52/5_mnist_full_connection_RANUM.py",
    "ID_53": "model_zoo/grist_models/ID_53/h02_20191644_RANUM.py",
    "ID_54": "model_zoo/grist_models/ID_54/h03_20191644_RANUM.py",
    "ID_55": "model_zoo/grist_models/ID_55/logistic_regression_RANUM.py",
    "ID_56": "model_zoo/grist_models/ID_56/01_Logistic_Regression_RANUM.py",
    "ID_57": "model_zoo/grist_models/ID_57/03_softmax_cost_RANUM.py",
    "ID_58": "model_zoo/grist_models/ID_58/test_RANUM.py",
    "ID_59": "model_zoo/grist_models/ID_59/code_06_RANUM.py",
    "ID_60": "model_zoo/grist_models/ID_60/simple_saveAndReloadModel_RANUM.py",
    "ID_61": "model_zoo/grist_models/ID_61/multi_layer_perceptron_RANUM.py",
    "ID_62": "model_zoo/grist_models/ID_62/denoising_RBM_RANUM.py",
    "ID_63": "model_zoo/grist_models/ID_63/main_RANUM.py"
}

package_info = {
    "ID_1": "tensorflow",
    "ID_2a": "tensorflow",
    "ID_2b": "tensorflow",
    "ID_3": "tensorflow",
    "ID_4": "torch",
    "ID_5": "torch",
    "ID_6": "tensorflow",
    "ID_7": "tensorflow",
    "ID_8": "tensorflow",
    "ID_9a": "tensorflow",
    "ID_9b": "tensorflow",
    "ID_10": "tensorflow",
    "ID_11a": "tensorflow",
    "ID_11b": "tensorflow",
    "ID_11c": "tensorflow",
    "ID_12": "torch",
    "ID_13": "torch",
    "ID_14": "tensorflow",
    "ID_15": "tensorflow",
    "ID_16a": "tensorflow",
    "ID_16b": "tensorflow",
    "ID_16c": "tensorflow",
    "ID_17": "tensorflow",
    "ID_18": "tensorflow",
    "ID_19": "tensorflow",
    "ID_20": "tensorflow",
    "ID_21": "tensorflow",
    "ID_22": "torch",
    "ID_23": "torch",
    "ID_24": "tensorflow",
    "ID_25": "tensorflow",
    "ID_26": "torch",
    "ID_27": "torch",
    "ID_28d": "tensorflow",
    "ID_28c": "tensorflow",
    "ID_28a": "tensorflow",
    "ID_28b": "tensorflow",
    # "ID_28e": "tensorflow",
    "ID_29": "tensorflow",
    "ID_30": "tensorflow",
    "ID_31": "tensorflow",
    "ID_32": "torch",
    "ID_33": "torch",
    "ID_34": "torch",
    # "ID_35a": "tensorflow",
    "ID_35b": "tensorflow",
    "ID_35a": "tensorflow",
    "ID_36a": "tensorflow",
    # "ID_36b": "tensorflow",
    "ID_36b": "tensorflow",
    "ID_37": "torch",
    # for ablation study, we construct 37z
    "ID_37z": "torch",
    "ID_38": "torch",
    # for ablation study, we construct 38z
    "ID_38z": "torch",
    "ID_39a": "tensorflow",
    # "ID_39b": "tensorflow",
    "ID_39b": "tensorflow",
    "ID_40": "tensorflow",
    "ID_41": "tensorflow",
    "ID_42": "tensorflow",
    "ID_43a": "tensorflow",
    # "ID_43b": "tensorflow",
    "ID_43b": "tensorflow",
    "ID_44": "tensorflow",
    "ID_45a": "tensorflow",
    "ID_45b": "tensorflow",
    "ID_46": "torch",
    "ID_47": "torch",
    "ID_48a": "tensorflow",
    "ID_48b": "tensorflow",
    "ID_49a": "tensorflow",
    # "ID_49b": "tensorflow",
    "ID_49b": "tensorflow",
    "ID_50": "tensorflow",
    "ID_51": "torch",
    "ID_52": "tensorflow",
    "ID_53": "torch",
    "ID_54": "torch",
    "ID_55": "tensorflow",
    "ID_56": "torch",
    "ID_57": "torch",
    "ID_58": "tensorflow",
    "ID_59": "tensorflow",
    "ID_60": "tensorflow",
    "ID_61": "tensorflow",
    "ID_62": "torch",
    "ID_63": "torch"
}

run_ids = []

# only for tensorflow models
inputs_outputs = {
    "ID_1": (['x', 'y', 'keep_prob'], ['cross_entropy', 'obj_function']),
    "ID_2a": (['x-input', 'y-input'], ['cost/cost', 'cost/obj_function']),
    "ID_2b": (['x-input', 'y-input'], ['cost/cost', 'cost/obj_function']),
    "ID_3": (['x', 'y'], ['cost', 'obj_function']),
    "ID_6": (['x', 'y', 'keep_prob'], ['cross_entropy']),
    "ID_7": (['x', 'y', 'keep_prob'], ['cross_entropy', 'obj_function']),
    "ID_8": (['x', 'y', 'keep_prob'], ['cross_entropy', 'obj_function']),
    "ID_9a": (['x', 'y'], ['loss', 'obj_function']),
    "ID_9b": (['x', 'y'], ['loss', 'obj_function']),
    "ID_10": (['x', 'y'], ['cross_entropy', 'obj_function']),
    # seems the DynamicStitch is still not supported by tf2onnx
    # "ID_11": (['x', 'y'], ['loss', 'obj_function', 'obj_grad']),
    "ID_11a": (['x', 'y'], ['loss', 'obj_function']),
    "ID_11b": (['x', 'y'], ['loss', 'obj_function']),
    "ID_11c": (['x', 'y'], ['loss', 'obj_function']),
    "ID_14": (['input_x', 'label'], ['loss']),
    "ID_15": (['X', 'S1', 'S2', 'y'], ['cost', 'err', 'obj_function']),
    "ID_16a": (['x', 'y'], ['loss', 'obj_function']),
    "ID_16b": (['x', 'y'], ['loss', 'obj_function']),
    "ID_16c": (['x', 'y'], ['loss', 'obj_function']),
    "ID_17": (['x', 'y'], ['cross_entropy', 'loss', 'accuracy', 'obj_function']),
    "ID_18": (['x', 'labels'], ['loss/cross_entropy', 'obj_function']),
    "ID_19": (['x_input', 'y'], ['cross_entropy', 'obj_var']),
    "ID_20": (['x', 'y'], ['cost', 'obj_var']),
    "ID_21": (['x', 'y'], ['cross_entropy', 'obj_function']),
    "ID_24": (['input/x-input', 'input/y-input'], ['cross_entropy', 'obj_function']),
    "ID_25": (['x', 'y'], ['cross_entropy', 'obj_var']),
    "ID_28d": (['features', 'labels'], ['loss']),
    "ID_28c": (['features', 'labels'], ['loss', 'obj_function']),
    "ID_28a": (['features', 'labels'], ['loss', 'obj_function']),
    "ID_28b": (['features', 'labels'], ['loss', 'obj_function']),
    # "ID_28e": (['features', 'labels'], ['loss', 'obj_function']),
    "ID_29": (['x', 'y_'], ['cross_entropy', 'obj_function']),
    "ID_30": (['x', 'y_'], ['cross_entropy', 'obj_function', 'obj_var']),
    "ID_31": (['x', 'z', 'keep_prob'], ['D_loss', 'obj_function']),
    # "ID_35a": (['x', 'eps'], ['cost', 'obj_function']),
    "ID_35b": (['x', 'eps'], ['cost', 'obj_function']),
    "ID_35a": (['x', 'eps'], ['cost', 'obj_function']),
    "ID_36a": (['x', 'eps'], ['cost', 'obj_function']),
    # "ID_36b": (['x', 'eps'], ['cost', 'obj_function']),
    "ID_36b": (['x', 'eps'], ['cost', 'obj_function']),
    "ID_39a": (['x', 'eps'], ['cost', 'obj_function']),
    # "ID_39b": (['x', 'eps'], ['cost', 'obj_function']),
    "ID_39b": (['x', 'eps'], ['cost', 'obj_function']),
    "ID_40": (['x', 'y_'], ['cross_entropy']),
    "ID_41": (['x', 'y_'], ['cross_entropy', 'obj_function']),
    "ID_42": (['x', 'y_'], ['loss/cross_entropy', 'obj_function']),
    "ID_43a": (['x', 'y', 'eps'], ['cost', 'obj_function']),
    # "ID_43b": (['x', 'y', 'eps'], ['cost', 'obj_function']),
    "ID_43b": (['x', 'y', 'eps'], ['cost', 'obj_function']),
    "ID_44": (['x', 'y_'], ['cross_entropy', 'obj_function']),
    "ID_45a": (['x', 'y_'], ['cross_entropy', 'obj_function']),
    "ID_45b": (['x', 'y_'], ['cross_entropy', 'obj_function']),
    "ID_48a": (['x', 'y_true', 'keep_prob'], ['cross_entropy', 'obj_function']),
    "ID_48b": (['x', 'y_true'], ['cross_entropy', 'obj_function']),
    "ID_49a": (['x', 'y', 'eps'], ['cost', 'obj_function']),
    # "ID_49b": (['x', 'y', 'eps'], ['cost', 'obj_function']),
    "ID_49b": (['x', 'y', 'eps'], ['cost', 'obj_function']),
    "ID_50": (['x', 'y_'], ['cross_entropy', 'obj_function']),
    "ID_52": (['x', 'y_'], ['cross_entropy', 'obj_function']),
    "ID_55": (['x', 'y', 'keep_prob'], ['cross_entropy', 'obj_function']),
    "ID_58": (['x', 'y_', 'keep_prob'], ['cross_entropy', 'obj_function']),
    "ID_59": (['x', 'y'], ['cross_entropy', 'obj_function']),
    "ID_60": (['x', 'y_'], ['cross_entropy', 'obj_function']),
    "ID_61": (['x', 'y_', 'keep_prob'], ['cross_entropy', 'obj_function'])
}

import sys
sys.path.append('.')
sys.path.append('..')
import os
import json
import shutil
import subprocess

from config import SERVER_PYTHON_PATH, timeout_for_grist_execution, proj_root

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
    """
        Copy .py script which has correponding _grist.py version as _RANUM.py, so that we can edit the model file
        Note that we copy from original script
    :return:
    """
    current_new_scripts = dict()
    for k, v in current_selected_scripts.items():
        dir_name = os.path.dirname(v)
        file_name = os.path.basename(v)
        kw_lists = ['_grist']
        for kw in kw_lists:
            if file_name.count(kw) > 0:
                file_name = file_name.replace(kw, '')
        assert os.path.exists(dir_name + '/' + file_name)
        new_file_name = file_name.replace('.', '_RANUM.')
        print(dir_name + '/' + new_file_name)
        print(os.path.exists(dir_name + '/' + new_file_name))
        if not os.path.exists(dir_name + '/' + new_file_name):
            # copy non-grist scripts
            shutil.copy(dir_name + '/' + file_name, dir_name + '/' + new_file_name)
            print('copy', dir_name + '/' + file_name, 'to', dir_name + '/' + new_file_name)
        current_new_scripts[k] = dir_name + '/' + new_file_name

    # one time use to create "current_new_scripts" variable
    print(json.dumps(current_new_scripts, indent=2))


def package_detect():
    """
        Detect what framework the script is based on
    :return:
    """
    torch_cnt = 0
    tf_cnt = 0

    pre_knowledge = {17: 'tensorflow'}

    package_info = dict()

    for id in current_new_scripts:
        is_torch = is_tensorflow = False
        with open(current_new_scripts[id], 'r') as f:
            code = f.read()
        if code.count('torch') > 0:
            is_torch = True
        if code.count('tensorflow') > 0:
            is_tensorflow = True
        if int(id[3:]) in pre_knowledge:
            knowledge = pre_knowledge[int(id[3:])]
            if knowledge == 'tensorflow':
                is_torch, is_tensorflow = False, True
            elif knowledge == 'torch':
                is_torch, is_tensorflow = True, False
        if is_torch and is_tensorflow:
            print(f'both torch and tf found in {id}: {current_new_scripts[id]}')
        elif not is_torch and not is_tensorflow:
            print(f'missing package in {id}: {current_new_scripts[id]}')
        else:
            torch_cnt += is_torch
            tf_cnt += is_tensorflow
            if is_torch: package_info[id] = 'torch'
            if is_tensorflow: package_info[id] = 'tensorflow'

    print('Torch Scripts:', torch_cnt, 'Tensorflow Scripts:', tf_cnt)
    print(json.dumps(package_info, indent=2))

def script_runner(white_list=[], black_list=[]):
    """
        Running the new script ended with _RANUM.py
    :param white_list:
    :param black_list:
    :return:
    """
    failed_list = []
    for id, script_path in current_new_scripts.items():
        num = id[3:]
        if len(white_list) > 0 and num not in white_list: continue
        if num in black_list: continue

        print(f'Saving for {id}')
        # now we can run it
        subproc = subprocess.Popen(f'PYTHONPATH=. timeout {timeout_for_grist_execution} {SERVER_PYTHON_PATH} {script_path}',
                                   shell=True, cwd=proj_root)
        outs, errs = subproc.communicate()
        print(f'Exit with code {subproc.returncode}')
        # print(outs)
        # print(errs, file=sys.stderr)

        if package_info[id] == 'tensorflow':
            # for tf models, we externally run tf2onnx to convert it to onnx
            print('  Convert to onnx...')
            subproc = subprocess.Popen(f'timeout {timeout_for_grist_execution} {SERVER_PYTHON_PATH} -m tf2onnx.convert '
                                       f'--opset 13 '
                                       f'--checkpoint model_zoo/grist_protobufs/{id}/model.ckpt.meta '
                                       f'--output model_zoo/grist_protobufs/{id}/model.onnx '
                                       f'--inputs {",".join([x + ":0" for x in inputs_outputs[id][0]])} '
                                       f'--outputs {",".join([x + ":0" for x in inputs_outputs[id][1]])}',
                                       shell=True, cwd=proj_root)
            outs, errs = subproc.communicate()
            print(f'  Conversion exit with code {subproc.returncode}')
            if subproc.returncode != 0:
                failed_list.append(id)

    print('Failed cases:', failed_list)


if __name__ == '__main__':
    # script_scan()
    # script_copy()
    # package_detect()
    script_runner(run_ids)

