import json
import matplotlib.pyplot as plt


def parameter_search():
    batch_size_params = [4, 8, 16, 32, 64]
    weight_decay_params = [0.0, 10e-3, 10e-6, 10e-9, 10e-12]
    num_hidden_units_params = [5, 10, 15, 20, 25]
    best_files = {4: None, 8: None, 16: None, 32: None, 64: None}
    best_file = None
    best_acc = 0
    for batch_size in batch_size_params:
        for weight_decay in weight_decay_params:
            for num_hidden_units in num_hidden_units_params:
                log_file = "3_mlp_%s_%d_%d.json" % (weight_decay, batch_size, num_hidden_units)
                with open("./logs/"+log_file) as json_data:
                    d = json.load(json_data)
                    if not best_file or d["best_acc"] > best_acc:
                        best_file = log_file
                        best_acc = d["best_acc"]
        best_files[batch_size] = best_file
    return best_files


def main():
    # from a2: Best batch size = 32 with file  3_mlp_1e-06_32_10.json
    # from a3: Best num_hidden_unit = 25 with file 3_mlp_1e-06_32_25.json
    # from a4: Best decay param = 1e-12 with file 3_mlp_1e-12_32_25.json
    log_files = a5()
    fig = plt.figure()
    best_batch_size = 0
    best_acc = 0
    for param, file in log_files.items():
        with open("./logs/"+file) as json_data:
            d = json.load(json_data)
            # Code for getting avgs
            # d["test_acc_avg"] = []
            # d["train_loss_avg"] = []
            # for i in range(0, 1000, 100):
            #     d["train_loss_avg"].append(sum(d["train_loss"][i:i+100])/100)
            #     d["test_acc_avg"].append(sum(d["test_acc"][i:i+100])/100)
            if d["best_acc"] > best_acc:
                best_acc = d["best_acc"]
                best_batch_size = param
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            print(file)
            epoch_range = [0, 99, 199, 299, 399, 499, 599, 699, 799, 899, 999]
            ax1.plot([d["epoch"][i]+1 for i in epoch_range], [d["test_acc"][i]
                     for i in epoch_range], label=str(param))
            ax2.plot([d["epoch"][i]+1 for i in epoch_range], [d["train_acc"][i]
                     for i in epoch_range], label=str(param))
            ax1.legend()
            ax2.legend()
            ax1.set_xlabel("EPOCHS")
            ax1.set_ylabel("Test Accuracy")
            ax2.set_xlabel("EPOCHS")
            ax2.set_ylabel("Train Accuracy")
            ax1.set_title("(a)")
            ax2.set_title("(b)")
    plt.show()
    print(best_batch_size)


def a2():
    batch_size_params = [4, 8, 16, 32, 64]
    weight_decay = 1e-06
    num_hidden_units = 10
    log_files = {batch_size: "3_mlp_%s_%d_%d.json" % (weight_decay, batch_size, num_hidden_units) for batch_size in batch_size_params}
    return log_files


def a3():
    batch_size = 32   # best from a2
    weight_decay = 1e-06
    num_hidden_units_params = [5, 10, 15, 20, 25]
    log_files = {num_hidden_units: "3_mlp_%s_%d_%d.json" % (weight_decay, batch_size, num_hidden_units) for num_hidden_units in num_hidden_units_params}
    return log_files


def a4():
    batch_size = 32   # best from a2
    weight_decay_params = [0.0, 1e-03, 1e-06, 1e-09, 1e-12]
    num_hidden_units = 25   # best from a3
    log_files = {weight_decay: "3_mlp_%s_%d_%d.json" % (weight_decay, batch_size, num_hidden_units) for weight_decay in weight_decay_params}
    return log_files


def a5():
    batch_size = 32   # best from a2
    weight_decay = 1e-06
    num_hidden_units = 10   # best from a3
    log_files = {weight_decay: "4_mlp_%s_%d_%d.json" % (weight_decay, batch_size, num_hidden_units)}
    return log_files


if __name__ == '__main__':
    # main()
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.plot([4, 8, 16, 32, 64], [2.05413842201, 3.73041439056, 6.41366624832, 11.476984024, 22.1030628681], label="Average time in milliseconds")
    ax2.plot([5, 10, 15, 20, 25], [11.6390998363, 12.2136061192, 11.3611791134, 11.3937628269, 11.4918627739], label="Average time in milliseconds")

    ax1.legend()
    ax2.legend()
    ax1.set_xlabel("Batch size")
    ax1.set_ylabel("Average time in milliseconds")
    ax2.set_xlabel("Number of neurons in hidden layer")
    ax2.set_ylabel("Average time in milliseconds")
    ax2.set_ylim(0, 23)
    ax1.set_ylim(0, 23)
    plt.show()
