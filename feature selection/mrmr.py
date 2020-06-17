import pandas as pd
import pymrmr
import xlwt
def mrmr_feature(csv_path,feature_number_list):

    df = pd.read_csv(csv_path)

    for feature_number in feature_number_list:
        result=pymrmr.mRMR(df,'MIQ',feature_number)
        book = xlwt.Workbook()
        sheet1 = book.add_sheet(u'sheet1',cell_overwrite_ok=True)
        for i in range(len(result)):
            name = result[i]
            for j in range(len(df[name])+1):
                if j==0:
                    sheet1.write(j,i,name)
                else:
                    sheet1.write(j,i,float(df[name][j-1]))