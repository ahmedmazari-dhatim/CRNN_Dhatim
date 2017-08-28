# -*- coding: utf-8 -*-
from __future__ import division
import pandas as pd
from subs_del_inser import levenshtein_ids


def get_char_accuracy():
    df = pd.read_csv('/home/ahmed/Pictures/cogedis/24072017/all+MAJ/all_clean_MAJ.csv', sep=',')
    df = df.astype(str)
    df = df.replace(['é', 'è', 'È', 'É'], 'e', regex=True)
    df = df.replace(['à', 'â', 'Â'], 'a', regex=True)
    substitution=0
    insertion=0
    deletion=0
    correct=0

    for i in range(len(df)):
        if df.manual_raw_value[i] != df.raw_value[i]:
            text = df.manual_raw_value[i]
            text2 = df.raw_value[i]
            x,y,z= levenshtein_ids(text,text2)
            insertion +=x
            deletion +=y
            substitution +=z

        else:
            correct +=1
    x = (1, 2, 3)
    y = (substitution, insertion, deletion)
    z=substitution+insertion+deletion
    sub_error= substitution/z

    inser_err=insertion/z
    deletion_err=deletion/z
    num_char=pd.Series(list(df.manual_raw_value.str.cat())).count()
    sub_error_all = substitution / num_char
    inser_err_all = insertion / num_char
    deletion_err_all = deletion / num_char
    acc=100*(1-(z/num_char))

    return acc,substitution,insertion,deletion,sub_error,inser_err,deletion_err,sub_error_all,inser_err_all,deletion_err_all

if __name__ == '__main__':
    acc, substitution, insertion, deletion,sub_error,inser_err,deletion_err,sub_error_all,inser_err_all,deletion_err_all=get_char_accuracy()
    print('accuracy ',acc)
    print('substitution ', substitution)
    print('insertion ', insertion)
    print('deletion ', deletion)
    print('substitution error rate ', sub_error)
    print('insertion error rate', inser_err)
    print('deletion error rate', deletion_err)



