# E:/pydata
# -*- coding: utf-8 -*-
# @Time : 2022/9/2 21:37
# @Author : Kaweh  Kaweh.jiang@miniso.com
# @Site : 
# @File : 034汇总记录.py
# @Software: PyCharm
from datetime import datetime
import xlwings as xw
import  pandas as pd
from pandas import Series,DataFrame
import time
from sqlalchemy import create_engine
def dataimportmysql(df,user,pwd,ip,database,table):
    user=user
    pwd=pwd
    ip=ip
    database=database
    df=df
    table=table
    yconnerct = create_engine(r'mysql+pymysql://'+user+':'+pwd+'@'+ip+'/'+database+'?charset=utf8')
    pd.io.sql.to_sql(df, table, yconnerct, schema=database, if_exists='append', index=False)
import datetime as dtt
startime=datetime.now()
star=datetime.now()
print('开始时间 %s' % startime)
app=xw.App(visible=False,add_book=False)
app.display_alerts = False # 警告关闭
app.screen_updating = False # 屏幕更新关闭
a=time.strftime('%Y%m%d', time.localtime())
ab=time.strftime('%Y-%m-%d', time.localtime())
# a=time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
###处理034
aaa=[]
startime=datetime.now()
list034=['1021','1022','1023','1027','8050']
for i in list034:
    print('%s    正在打开%s 仓库龄数量' % (time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime()),i))
    wb034=app.books.open(r'\\10.130.182.40\海外业务支持部\基础数据\01-源数据\源数据'+a+'\ZEWM034-'+i+a+'.xlsx')
    sheet=wb034.sheets[0]
    x,y=sheet.used_range.shape
    res=pd.DataFrame(sheet.range((2,1),(x,y)).value,columns=Series(sheet.range((1,1),(1,y)).value))
    res034=res[['货品代码','货位','基本数量','收货日期和时间','保质期','剩余保质期天数']]
    res034['wrh_id']=i
    aaa.append(res034)
    # res034.rename(columns={'货品代码':'商品代码'},inplace=True)
    wb034.close() # 关闭文件
    print('%s    %s 仓计算完毕' % (time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime()), i))
res034=pd.concat(aaa)
res034['ds']=ab
res034.rename(columns={'货品代码':'sku_id','货位':'no','基本数量':'qty','收货日期和时间':'receive_time','保质期':'expireday','剩余保质期天数':'remain_day'},inplace=True)
print('%s    正在导入mysql库' % time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime()))
dataimportmysql(res034,"kylin_wms","35mhy7AwzZxy3eHM","10.231.3.128:3306","kylin_wms","zewm034")
print('%s    导入MySQL库结束' % time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime()))