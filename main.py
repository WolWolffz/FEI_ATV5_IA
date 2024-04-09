import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
import time

print('Carregando Arquivo de teste')
arquivo = np.load('teste5.npy')
x = arquivo[0]
y = np.ravel(arquivo[1])

#nessa função, é retornado um vetor com os 3 melhores resultados (leia-se, com menor erro) e outro com todos os resultados
def ativarRegressor(iter,changeLimit,attempts,layers):
    start_total = time.process_time()
    #regressor. 'max_iter' é o número máximo de iterações que serão realizadas, 'n_iter_no_change' é o máximo de iterações sem mudança no erro para que a regressão pare
    regr = MLPRegressor(hidden_layer_sizes=(layers),
                        max_iter=iter,
                        activation='relu', #{'identity', 'logistic', 'tanh', 'relu'},
                        solver='adam',
                        learning_rate = 'adaptive',
                        n_iter_no_change=changeLimit)

    print('Treinando RNA')
    y_est = []
    loss = []
    loss_curves = []
    for i in range(0,attempts):
        print("Tentativa {}: ".format(i))
        
        start = time.process_time()
        regr = regr.fit(x,y)
        y_est.append(regr.predict(x))
        end = time.process_time()
        
        print("Melhor erro: {}".format(regr.best_loss_))
        loss.append(regr.best_loss_)
        loss_curves.append(regr.loss_curve_)
        print("Tempo: {}s".format(end-start))
    
    end_total = time.process_time()
    elapsed = end_total - start_total
    print("Tempo para todas as tentativas: {}s".format(elapsed))  # will print the elapsed time in seconds
    return y_est, loss, loss_curves, elapsed

def plotAll(y_est, curves):
    plt.figure(figsize=[14,7])

    #plot curso original
    plt.subplot(1,3,1)
    plt.plot(x,y)

    #plot aprendizagem
    plt.subplot(1,3,2)
    plt.plot(curves)

    #plot regressor
    plt.subplot(1,3,3)
    plt.plot(x,y,linewidth=1,color='green')
    plt.plot(x,y_est,linewidth=2,color='red')

    plt.show()

#o código abaixo executa três diferentes configurações de camadas escondidas do regressor, com 10k eras cada e 10 diferentes iterações para cada configuração, plotando tudo no final
#no console, é printado o melhor erro de cada tentativa para cada configuração
#após todas as iterações, também é printado a média e e desvio padrão do melhor erro dentre as dez iterações de cada configuração
#ativarRegressor(limite_total_eras, limite_sem_mudança, qtd_iterações, camadas_escondidas)
y_est1, loss1, curves1, elapsed1 = ativarRegressor(100000,2000,10,[2])
y_est2, loss2, curves2, elapsed2 = ativarRegressor(100000,2000,10,[6,2])
y_est3, loss3, curves3, elapsed3 = ativarRegressor(100000,2000,10,[10,6,4,2])

print("""
      Configuração 1:
        média dos erros: {}
        desvio padrão: {}
        melhor erro: {}
        tempo: {}s
      Configuração 2:
        média dos erros: {}
        desvio padrão: {}
        melhor erro: {}
        tempo: {}s
      Configuração 3:
        média dos erros: {}
        desvio padrão: {}
        melhor erro: {}
        tempo: {}s
      """.format(np.mean(loss1), np.std(loss1) ,np.amin(loss1), elapsed1,
                 np.mean(loss2), np.std(loss2) ,np.amin(loss2), elapsed2,
                 np.mean(loss3), np.std(loss3) ,np.amin(loss3), elapsed3))

plotAll(y_est1.pop(loss1.index(np.amin(loss1))),curves1.pop(loss1.index(np.amin(loss1))))
plotAll(y_est2.pop(loss2.index(np.amin(loss2))),curves2.pop(loss2.index(np.amin(loss2))))
plotAll(y_est3.pop(loss3.index(np.amin(loss3))),curves3.pop(loss3.index(np.amin(loss3))))