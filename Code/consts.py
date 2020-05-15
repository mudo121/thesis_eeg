DEVICES_MUSE_MONITOR = "muse_monitor"
DEVICES_OPEN_BCI = "open_bci"
DEVICES_MUSE_LSL = "muse-lsl"
DEVICES_NEUROSCAN = "neuroscan"

SUPPORTED_DEVICES = [DEVICES_MUSE_MONITOR,
                     DEVICES_OPEN_BCI,
                     DEVICES_MUSE_LSL]

TARGET_AWAKE = "AWAKE"
TARGET_FATIGUE = "FATIGUE"
TARGET_NORMAL = "NORMAL"
TARGET_UNLABELED = "UNLABELED"

TARGET_LABEL_DICT = {TARGET_AWAKE : 0,
                    TARGET_FATIGUE : 1,
                    TARGET_NORMAL : 2,
                    TARGET_UNLABELED : 3}


INDEX_BANDPOWER_LIST = 'bandpower_list'
INDEX_BANDPOWER_UPPER_ENVELOPE_LIST = 'bandpower_upper_envelope_list'
INDEX_BANDPOWER_LOWER_ENVELOPE_LIST = 'bandpower_lower_envelope_list'

INDEX_MEAN_BANDPOWER_LIST = 'mean_bandpower_list'
INDEX_MEAN_BANDPOWER_UPPER_ENVELOPE_LIST = 'mean_bandpower_upper_envelope_list'
INDEX_MEAN_BANDPOWER_LOWER_ENVELOPE_LIST = 'mean_bandpower_lower_envelope_list'

INDEX_STD_DEV_BANDPOWER_LIST = 'std_dev_bandpower_list'
INDEX_STD_DEV_BANDPOWER_UPPER_ENVELOPE_LIST = 'std_dev_bandpower_upper_envelope_list'
INDEX_STD_DEV_BANDPOWER_LOWER_ENVELOPE_LIST = 'std_dev_bandpower_lower_envelope_list'

# Used to create dynamically the columns for a feature dataframe (in the function: 'createNiceSeriesOfDataframes()')
FEATURE_DATAFRAME_COLUMNS = [INDEX_MEAN_BANDPOWER_LIST, INDEX_MEAN_BANDPOWER_LOWER_ENVELOPE_LIST, INDEX_MEAN_BANDPOWER_UPPER_ENVELOPE_LIST,
                            INDEX_STD_DEV_BANDPOWER_LIST, INDEX_STD_DEV_BANDPOWER_LOWER_ENVELOPE_LIST, INDEX_STD_DEV_BANDPOWER_UPPER_ENVELOPE_LIST]