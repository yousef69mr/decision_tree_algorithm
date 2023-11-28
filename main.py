from sklearn.model_selection import train_test_split
from sklearn import tree
import pandas as pd
import matplotlib.pyplot as plt


def experiment(dataset, test_sizes, random_states):
    y = dataset.iloc[:, -1]
    x = dataset.iloc[:, :-1]
    accuracy_means = list()
    tree_nodes_means = list()
    accuracies = list()
    tree_nodes = list()

    if len(test_sizes) > 1:
        for test_size in test_sizes:
            # scores is list that store all the scores of single training set
            scores = list()
            # nodes is list that store all the no. of nodes in tree of single training set
            nodes = list()
            print("Test Size : ", test_size)
            for state in random_states:
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=state)
                # print(x_train, x_test, y_train, y_test)

                # plt.figure(figsize=(10, 10))

                decision_tree = tree.DecisionTreeClassifier(criterion="entropy")
                decision_tree = decision_tree.fit(x_train, y_train)
                # tree.plot_tree(decision_tree)
                # plt.title("Random State : %d\n Test Size : " % state + str(test_size))
                score = decision_tree.score(x_test, y_test)
                nodes.append(decision_tree.tree_.node_count)
                # print(decision_tree.tree_.node_count)
                # print(decision_tree.get_n_leaves())
                scores.append(score)
                print("State : ", state, " , Accuracy : ", score, " ,No. of Nodes : ", decision_tree.tree_.node_count)
            accuracy_means.append(sum(scores)/len(scores))
            tree_nodes_means.append(sum(nodes)/len(nodes))
            tree_nodes.append(list([min(nodes), max(nodes)]))
            accuracies.append(list([min(scores), max(scores)]))
            # plt.show()
        print(accuracies)
        print(tree_nodes)
        print(accuracy_means)
        print(tree_nodes_means)
        return accuracies, accuracy_means, tree_nodes, tree_nodes_means

    else:
        test_size = test_sizes[0]
        scores = list()
        nodes = list()
        for state in random_states:

            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=state)
            # print(x_train, x_test, y_train, y_test)

            # plt.figure(figsize=(10, 10))

            decision_tree = tree.DecisionTreeClassifier(criterion="entropy")
            decision_tree = decision_tree.fit(x_train, y_train)
            # tree.plot_tree(decision_tree)
            # plt.title("Random State : %d\n Test Size : " % state + str(test_size))
            # print(test_size)
            score = decision_tree.score(x_test, y_test)
            nodes.append(decision_tree.tree_.node_count)
            scores.append(score)
            print("State : ", state, " , Accuracy : ", score, " ,No. of Nodes : ", decision_tree.tree_.node_count)
        # plt.show()
        accuracy_means.append(sum(scores) / len(scores))
        tree_nodes_means.append(sum(nodes) / len(nodes))
        tree_nodes.append(list([min(nodes), max(nodes)]))
        accuracies.append(list([min(scores), max(scores)]))
        print(accuracies)
        print(tree_nodes)
        print(accuracy_means)
        print(tree_nodes_means)

        return accuracies, accuracy_means, tree_nodes, tree_nodes_means


def run():
    dataset = pd.read_csv('BankNote_Authentication.csv')
    # print(dataset)
    # random_states = random.sample(range(0, 100), 5)
    random_states = list([38, 53, 22, 69, 71])
    print(random_states)
    test_sizes = list([0.7, 0.6, 0.5, 0.4, 0.3])
    train_sizes = test_sizes[::-1]
    print(train_sizes)
    print("Experiment 1 ")

    experiment(dataset, [.75], random_states)
    print("Experiment 2 ")
    accuracies, accuracy_means, tree_nodes, tree_nodes_means = experiment(dataset, test_sizes, random_states)
    plt.plot(train_sizes, accuracies, marker='o')
    plt.title("Plot 1")
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.grid(color='green', linestyle='--', linewidth=0.5)
    plt.show()

    plt.plot(train_sizes, tree_nodes, marker='o')
    plt.title("Plot 2")
    plt.xlabel("Training Set Size")
    plt.ylabel("Number of Nodes")
    plt.grid(color='green', linestyle='--', linewidth=0.5)
    plt.show()


if __name__ == '__main__':
    run()
