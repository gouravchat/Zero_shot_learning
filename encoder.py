

from sklearn.preprocessing import LabelEncoder
import pandas as pd




def labelEncode(df, le = None):
 
    if not le:
        le = LabelEncoder()
        df['category'] = le.fit_transform(df['category'])
        return le,df 
    else:
        df['category'] = le.transform(df['category'])
        return df



if __name__ =="__main__" :

    train_data = pd.read_excel("train_data.xlsx",engine = "openpyxl")
    print("Data loaded............................")
    train_data = train_data.dropna()
    train_data = train_data.sample(5000)
    encoder,train_data = labelEncode(train_data,le=None)

    classes = encoder.classes_.tolist()

    print(classes)
    print("Number of classes ... ",len(classes))

    # test Data 

    test_data = pd.read_excel("test_data_with_zsl_labels.xlsx",engine = "openpyxl")
    print("Data loaded............................")
    test_data = test_data.dropna()
    test_data = test_data.sample(1000)
    test_data = labelEncode(test_data,le=encoder)
    print(test_data)



 