import pandas as pd
import numpy as np
import os.path

def load_dataset():
    headers = ["word_freq_make", "word_freq_address", "word_freq_all", "word_freq_3d", "word_freq_our", "word_freq_over", "word_freq_remove", "word_freq_internet", "word_freq_order", "word_freq_mail", "word_freq_receive", "word_freq_will", "word_freq_people", "word_freq_report", "word_freq_addresses", "word_freq_free", "word_freq_business", "word_freq_email", "word_freq_you", "word_freq_credit", "word_freq_your", "word_freq_font", "word_freq_000", "word_freq_money", "word_freq_hp", "word_freq_hpl", "word_freq_george", "word_freq_650", "word_freq_lab", "word_freq_labs", "word_freq_telnet", "word_freq_857", "word_freq_data", "word_freq_415", "word_freq_85", "word_freq_technology", "word_freq_1999", "word_freq_parts", "word_freq_pm", "word_freq_direct", "word_freq_cs", "word_freq_meeting", "word_freq_original", "word_freq_project", "word_freq_re", "word_freq_edu", "word_freq_table", "word_freq_conference", "char_freq_;", "char_freq_(", "char_freq_[", "char_freq_!", "char_freq_$", "char_freq_#", "capital_run_length_average", "capital_run_length_longest", "capital_run_length_total", "spam"]

    dirname = os.path.abspath(os.path.dirname(__file__))
    data_path = os.path.join(dirname, "spam.data")
    test_train_split_path = os.path.join(dirname, "spam.traintest.txt")

    df = pd.read_csv(data_path, delim_whitespace=True, names=headers)
    test_train_split_ind = np.loadtxt(test_train_split_path)

    X = df.drop('spam', axis=1)
    y = df.spam.values

    y[y == 0] = -1

    train_mask, test_mask = (test_train_split_ind == 0), (test_train_split_ind == 1)

    X_train, X_test, y_train, y_test = X[train_mask], X[test_mask], y[train_mask], y[test_mask]

    return X_train, X_test, y_train, y_test
