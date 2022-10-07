def compare_metrics (model, name_dataset1, X, y, name_dataset2,  X2, y2 ):
    '''
    Take in a  model and compare the  performance metrics of  Train, Evaluate and Test (only 2).
    model: the model that you want to compare
    name_dataset1 : type :train, validate or  test. Select one, STRING
    X: df test, validate or test
    y: df test, validate or test
    name_dataset2: type :train, validate or  test. Select one, STRING
    X2: df2 test, validate or test
    y2: df2 test, validate or test
    
    Example:
    compare_metrics(logit2,'Train',X_train, y_train,'Test', X_test, y_test)
    '''
    
    if name_dataset1.lower() != "train" and name_dataset1.lower() != "validate" and name_dataset1.lower() != "test" :
        return print("incorrect name")
    if name_dataset2.lower() != "train" and name_dataset2.lower() != "validate" and name_dataset2.lower() != "test" :
        return print("incorrect name")
    #prediction
    pred_1 = model.predict(X)
    pred_2 = model.predict(X2)

    #score = accuracy
    acc_1 = model.score(X, y)
    acc_2 = model.score(X2, y2)


    #conf Matrix
    #model 1
    conf_1 = confusion_matrix(y, pred_1)
    mat_1 =  pd.DataFrame ((confusion_matrix(y, pred_1 )),index = ['actual_dead','actual_survived'], columns =['pred_dead','pred_survived' ])
    rubric_df = pd.DataFrame([['TN', 'FP'], ['FN', 'TP']], columns=mat_1.columns, index=mat_1.index)
    cf_1 = rubric_df + ' : ' + mat_1.values.astype(str)
    
    #model2
    conf_2 = confusion_matrix(y2, pred_2)
    mat_2 =  pd.DataFrame ((confusion_matrix(y2, pred_2 )),index = ['actual_dead','actual_survived'], columns =['pred_dead','pred_survived' ])
    cf_2 = rubric_df + ' : ' + mat_2.values.astype(str)
    #model 1
    #assign the values
    tp = conf_1[1,1]
    fp = conf_1[0,1] 
    fn = conf_1[1,0]
    tn = conf_1[0,0]

    #calculate the rate
    tpr_1 = tp/(tp+fn)
    fpr_1 = fp/(fp+tn)
    tnr_1 = tn/(tn+fp)
    fnr_1 = fn/(fn+tp)

    #model 2
    #assign the values
    tp = conf_2[1,1]
    fp = conf_2[0,1] 
    fn = conf_2[1,0]
    tn = conf_2[0,0]

    #calculate the rate
    tpr_2 = tp/(tp+fn)
    fpr_2 = fp/(fp+tn)
    tnr_2 = tn/(tn+fp)
    fnr_2 = fn/(fn+tp)

    #classification report
    #model1
    clas_rep_1 =pd.DataFrame(classification_report(y, pred_1, output_dict=True)).T
    clas_rep_1.rename(index={'0': "dead", '1': "survived"}, inplace = True)

    #model2
    clas_rep_2 =pd.DataFrame(classification_report(y2, pred_2, output_dict=True)).T
    clas_rep_2.rename(index={'0': "dead", '1': "survived"}, inplace = True)
    print(f'''
    ******    {name_dataset1}       ******                              ******     {name_dataset2}    ****** 
       Overall Accuracy:  {acc_1:.2%}              |                Overall Accuracy:  {acc_2:.2%}  
                                                
    True Positive Rate:  {tpr_1:.2%}               |          The True Positive Rate:  {tpr_2:.2%}  
    False Positive Rate:  {fpr_1:.2%}              |          The False Positive Rate:  {fpr_2:.2%} 
    True Negative Rate:  {tnr_1:.2%}               |          The True Negative Rate:  {tnr_2:.2%} 
    False Negative Rate:  {fnr_1:.2%}              |          The False Negative Rate:  {fnr_2:.2%}
    _________________________________________________________________________________
    ''')
    print('''
    Positive =  'survived'
    Confusion Matrix
    ''')
    cf_1_styler = cf_1.style.set_table_attributes("style='display:inline'").set_caption(f'{name_dataset1} Confusion Matrix')
    cf_2_styler = cf_2.style.set_table_attributes("style='display:inline'").set_caption(f'{name_dataset2} Confusion Matrix')
    space = "\xa0" * 50
    display_html(cf_1_styler._repr_html_()+ space  + cf_2_styler._repr_html_(), raw=True)
    print('''
    ________________________________________________________________________________
    
    Classification Report:
    ''')
    clas_rep_1_styler = clas_rep_1.style.set_table_attributes("style='display:inline'").set_caption(f'{name_dataset1} Classification Report')
    clas_rep_2_styler = clas_rep_2.style.set_table_attributes("style='display:inline'").set_caption(f'{name_dataset2} Classification Report')
    space = "\xa0" * 45
    display_html(clas_rep_1_styler._repr_html_()+ space  + clas_rep_2_styler._repr_html_(), raw=True)