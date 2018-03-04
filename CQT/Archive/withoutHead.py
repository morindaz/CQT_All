# coding=utf-8
import time
import glob
import pandas as pd
#这里的C数据和R相反
answerR = "E:\\pingan\\dataset\\newFeature\\answer_C" #492个数据，withouthead
answerC = "E:\\pingan\\dataset\\newFeature\\answer_R" #244个数据
answerI = "E:\\pingan\\dataset\\newFeature\\answer_I" #478个数据
base = answerI
csvx_list = glob.glob(base+"\\"+'*.csv')
print('总共发现%s个CSV文件'% len(csvx_list))
time.sleep(2)
print('正在处理............')
df = pd.DataFrame()
for i in csvx_list:
    df_c = pd.read_csv(i, sep=',', header=0)
    #    print(df_c['video_name'].tolist())
    # fr = i.values
    # print df_c
    df = df.append(df_c)
    #print df
    print('写入成功！')
output_Archive = pd.DataFrame(df)
output_Archive.to_csv("base"+'.csv')
print('写入完毕！')
print('3秒钟自动关闭程序！')
time.sleep(3)