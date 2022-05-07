from visualize import plot_loss, plot_acc
import matplotlib.pyplot as plt
# with open("./output/raf_gabor_history_array",'r') as his:
#     # plot_loss(his, "RAF")
#     # plot_acc(his, "RAF")
#     plt.plot(his['loss'])
with open(input("./output/raf_gabor_history_array", "r")) as his:
        plt.plot(his['loss'])