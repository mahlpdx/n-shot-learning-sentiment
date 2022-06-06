import pickle
from pathlib import Path

from dataloading import read_data, get_test_data
from dataprocessing import sentence_cleaning
from networkcompare import three_class_network_compare, twelve_class_network_compare
from networkanalysis import run_analysis


# Run Pipeline To See Network Comparisons and save analysis data as pickle
def networks_pipeline():
    # Load All Data
    main_data = read_data(Path(__file__).parent.parent / "data/Sentences.tsv")
    # Preprocess all data
    clean_data = sentence_cleaning(main_data)
    test_data = get_test_data(clean_data)
    # Do 3 class runs
    three_analysis_data = three_class_network_compare(clean_data, test_data)
    run_analysis(three_analysis_data)
    # Do 12 class Run
    twelve_analysis_data = twelve_class_network_compare(clean_data, test_data)
    run_analysis(twelve_analysis_data)
    print()

if __name__ == "__main__":
    networks_pipeline()
