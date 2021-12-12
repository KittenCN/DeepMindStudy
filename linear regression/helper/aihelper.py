import matplotlib.pyplot as plt
def show_data(x_data=[1, 2, 3], y_data=[1, 2, 3]):
    plt.scatter(x=x_data, y=y_data, color='k')
    plt.show()

def show_data_cost(x_data=[1, 2, 3], y_data=[1, 2, 3], prediction_data=[1, 2, 3], loss=0, use_gpu=1):
    plt.cla()
    if use_gpu:
        x_data = x_data.cpu().detach().numpy()
        y_data = y_data.cpu().detach().numpy()
        prediction_data = prediction_data.cpu().detach().numpy()
        loss = loss.cpu().detach().numpy()
    plt.scatter(x_data, y_data, c='r')
    plt.scatter(x_data, prediction_data, c='b')
    # plt.plot(x_data, prediction_data,'r-',lw=5)
    # plt.plot(x_data, prediction_data, color='k')
    plt.text(0.5,0,'Loss=%.4f'%loss, fontdict={'size':20,'color':'red'})
    plt.pause(0.1)#画的图只存在0.1秒