from tools import process_data, standardize_response_100ms, fra_plot

if __name__ == '__main__':
    # normal = process_data("./data/states/raw_data.pkl", raw_response)
    standardized_100ms = process_data("./data/states/standardized_100ms.pkl", standardize_response_100ms)
    # min_normalized = process_data("./data/states/min_normalized.pkl",
    #                               min_normalized_response_per_trigger)
    # min_normalized_across_all_stimuli = process_data("./data/states/min_normalized_across_all_stimuli.pkl",
    #                                                  min_normalized_response_per_neuron)
    fra_plot(standardized_100ms, 2, 45)
    # print(standardized_100ms[4000][0])
