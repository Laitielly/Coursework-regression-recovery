import statsmodels.api as sm
import matplotlib.pyplot as plt

def Make_Files(source, out1, out2):
    results = []
    file = open(source)
    file.readline()
    for i in file:
        row = list(map(float, i.split(',')))
        results.append(row)

    file.close()
    _len = len(results)
    __len = len(results[0])

    m = [0] * _len
    for j in range(_len):
        m[int(results[j][-1]) + 1] += 1

    f = open(out1, "w")
    f1 = open(out2, "w")
    for i in range(__len):
        m[i] = int(m[i] / 6) - 1

    for i in results:
        line = ''
        if (m[int(i[-1]) + 1]):
            line += str(i[0])
            for j in range(1, __len):
                line += ',' + str(i[j])
            f1.write(line + "\n")
            m[int(i[-1]) + 1] -= 1
        else:
            line += str(i[0])
            for j in range(1, __len):
                line += ',' + str(i[j])
            f.write(line + "\n")

    f.close()
    f1.close()

def Take_Data(source):
    results = []
    file = open(source)
    for i in file:
        i = '1, ' + i
        row = list(map(float, i.split(',')))
        results.append(row)

    file.close()
    _len = len(results)
    __len = len(results[0])

    Xi = [[]] * _len
    Yi = [[]] * _len
    for i in range(_len):
        Xi[i] = results[i][0:-1:]
        Yi[i] = results[i][-1]
    return (Xi, Yi)

def Make_Regression(Xi, Yi, _len):
    model = sm.OLS(Yi, Xi).fit()
    return (model.predict(Xi), model.params, model)

def SSE_SSR_SST(Yi, pred):
    SSE, Ym = 0, 0
    _len = len(Yi)
    remains = [[]] * _len
    for i in range(_len):
        SSE += (Yi[i] - pred[i]) ** 2
        remains[i] = Yi[i] - pred[i]
        Ym += Yi[i]

    Ym /= _len
    SSR = sum((pred[i] - Ym) ** 2 for i in range(_len))
    SST = SSE + SSR
    error = (SSE / (_len - 2)) ** (1 / 2)
    r2 = SSR/SST

    return (SSR, SST, SSE, r2, error, remains, Ym)

def Write_to_Res(res, predicts, r2, error, coefficient, SST, SSR, Yi, out1):
    out = out1 + " results:\nFiles: " + out1 + "\n\n"
    res.write(out)

    if (len(coefficient)):
        w = "Regression coefficients:\n" + str(coefficient) + "\n"
        res.write(w)

    #SSE -> SSR -> SST
    res.write("The standard error of the estimate ")
    res.write(str(round(error, 4)))
    res.write("\nSum of squared errors is equal to ")
    res.write(str(round(SST - SSR, 4)))
    res.write("\nSum of regression squares is equal to ")
    res.write(str(round(SSR, 4)))
    res.write("\nTotal sum of squares is equal to ")
    res.write(str(round(SST, 4)))
    res.write("\nDetermination coefficient is ")
    res.write(str(round(r2, 4)))
    res.write("\n")

    true, false = 0, 0
    _len = len(Yi)
    predicts = [round(i) for i in predicts]
    for i in range(_len):
        if (predicts[i] > 6 or predicts[i] < -1):
            if (predicts[i] > 6):
                predicts[i] = 6
            else:
                predicts[i] = -1
        if (Yi[i] == predicts[i]):
            true += 1
            continue
        false += 1

    res.write("The percentage of matches without errors is ")
    res.write(str(round(true / (true + false) * 100, 2)))
    res.write("%\n")

def DrawPlot(X, Y, namey, namex):
    plt.scatter(X, Y)
    plt.xlabel(namex)
    plt.ylabel(namey)

def DistributionOfResiduals(file, remains):
    quantile = FreeQuantile(file)
    remains.sort()
    DrawPlot(quantile, remains, 'quantile', 'sort remains')
    plt.show()

def FreeQuantile(source):
    file = open(source, "r")
    results = []
    for i in file:
        results.append(i)

    file.close()
    results = [float(i) for i in results]
    return results

def DWstat(remains):
    _len = len(remains)
    sumDW, sumsqDW = 0, (remains[0]) ** 2
    for i in range(1, _len):
        sumDW += remains[i - 1] * remains[i]
        sumsqDW += (remains[i]) ** 2
    DW = 2 - 2 * (sumDW / sumsqDW)
    return DW

def MSE_MSR_MST(SSE, SSR, _len, k):
    MSR = SSR / k
    MSE = SSE / (_len - k - 1)
    return MSR / MSE, MSR, MSE

def Trustint(coefficient, _len, __len, Xi, SSE, file, St):
    Syx = (SSE/(_len - 2)) ** (0.5)
    x = [0] * __len
    for j in range(__len):
        for i in range(_len):
            x[j] += Xi[i][j]
        x[j] /= _len

    SXX = [0] * __len
    for i in range(__len):
        b = 0
        for j in range(_len):
            b += (Xi[j][i] - x[i])**2
        SXX[i] = b ** (0.5)

    mu = [0] * __len
    for i in range(__len):
        if SXX[i] != 0:
            mu[i] = Syx / SXX[i]
        else:
            mu[i] = 0

    # t = [0] * __len
    # for i in range(__len):
    #     if mu[i] != 0:
    #         t[i] = round(coefficient[i] / mu[i])
    #     else:
    #         t[i] = 0

    a, b = 0, 0
    line = ''
    indepen = 'Регрессионные коэффициенты: '
    for i in range(1, __len):
        a = coefficient[i] + (St * mu[i])
        b = coefficient[i] - (St * mu[i])
        line += str(a) + ',' + str(b) + '\n'

        if (a > 0 and b < 0 or a < 0 and b > 0):
            indepen += str(i + 1) + ' '

    f = open(file, "w")
    f.write(line)
    f.write("\n")
    f.write(indepen)
    f.close()

def WriteTrainResults(res, MSR, MSE, F, DW, file):
    res.write("Mean square due to regression is equal to ")
    res.write(str(round(MSR, 4)))
    res.write("\nError variance is equal to ")
    res.write(str(round(MSE, 4)))
    res.write("\nTest F-statistics is equal to ")
    res.write(str(round(F, 4)))
    res.write("\nCoefficient of Durbin-Watson statistics is equal to ")
    res.write(str(round(DW, 4)))
    res.write("\nConfidence intervals are recorded in the file ")
    res.write(file)
    w = '\n\n' + '-' * 60 + '\n'
    res.write(w)

def main():
    file = "ObDataSet.csv"
    file1 = "ObData_train.csv"
    file2 = "ObData_test.csv"
    file3 = "quantile.csv"
    file4 = "resultT.csv"
    res = open("results.txt", "w")
    St = 1.961328292
    # Make_Files(file, file1, file2)

    Xi,Yi = Take_Data(file1)
    _len = len(Yi)
    __len = len(Xi[0])
    predicted, coefficient, model = Make_Regression(Xi, Yi, _len)
    SSR, SST, SSE, r2, error, remains, Ym = SSE_SSR_SST(Yi, predicted)
    Write_to_Res(res, predicted, r2, error, coefficient, SST, SSR, Yi, file1)
    #DrawPlot(predicted, remains, 'Yi', 'remains')
    #plt.show()
    # x = list(map(list, zip(*Xi)))
    # count = 1
    # for j in range(7):
    #     for i in range(4):
    #         if count <= 27:
    #             plt.subplot(2, 2, i + 1)
    #             namex = 'X' + str(count)
    #             DrawPlot(x[count], remains, 'remains', namex)
    #             count += 1
    #     plt.show()
    #DistributionOfResiduals(file3, remains)
    DW = DWstat(remains)
    F, MSR, MSE = MSE_MSR_MST(SSE, SSR, _len, __len - 1)
    Trustint(coefficient, _len, __len, Xi, SSE, file4, St)
    WriteTrainResults(res, MSR, MSE, F, DW, file4)

    Xi, Yi = Take_Data(file2)
    predicted = model.predict(Xi)
    SSR, SST, SSE, r2, error, remains, Ym = SSE_SSR_SST(Yi, predicted)
    Write_to_Res(res, predicted, r2, error, [], SST, SSR, Yi, file2)
    res.close()

main()