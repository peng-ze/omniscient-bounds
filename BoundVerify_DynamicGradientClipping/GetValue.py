# -*- coding: utf-8 -*-
import sys

def get_sp(log_file):
    findtxt1 = 'Acc-train: '
    findtxt2 = 'Acc-test: '
    findtxt3 = 'Grad-Var: '
    findtxt4 = 'Grad-Norm  '
    m1 = len(findtxt1)
    m2 = len(findtxt2)
    m3 = len(findtxt3)
    m4 = len(findtxt4)
    
    tr,ts,var,norm = [],[],[],[]
    with open(log_file, "r", encoding='utf-8') as f:
        lines=f.readlines()        
        for line in lines:
            if line != '\n':
                txt = str(line)
                if txt.find("Variance") != -1:
                    break
                if txt.find(findtxt1) == -1:
                    continue
                pos1 = txt.find(findtxt1)
                pos2 = txt.find(findtxt2)
                pos3 = txt.find(findtxt3)
                pos4 = txt.find(findtxt4)
                target = txt[pos1+m1:pos2]
                value = [s for s in target.split(' ')]
                tr.append('%.3f'%float(value[0]))
                target = txt[pos2+m2:pos3]
                value = [s for s in target.split(' ')]
                ts.append('%.3f'%float(value[0]))
                target = txt[pos3+m3:pos4]
                value = [s for s in target.split(' ')]
                var.append('%.3f'%float(value[0]))
                target = txt[pos4+m4:]
                value = [s for s in target.split(' ')]
                norm.append('%.3f'%float(value[0]))
             
    return tr, ts, var, norm


if __name__=='__main__':  
    log_file1 = sys.argv[1]
    name = str(sys.argv[2])+'.txt'
    tr, ts, var, norm = get_sp(log_file1)
    c_num = len(tr)
    length = len(tr)
    fo = open(name, 'w', encoding="utf8")
    fo.write('epoch'+' '+'TrAcc'+' '+'TsAcc'+' '+'GVar'+' '+'Gnorm')
    fo.write('\n')
    for i in range(int(c_num)):
        if i < len(tr):
            fo.write(str(i)+' '+tr[i]+' '+ts[i]+' '+var[i]+' '+norm[i])
        else:
            fo.write(str(i)+' '+tr[length-1]+' '+ts[length-1]+' '+var[length-1]+' '+norm[length-1])
        if i!=int(c_num)-1:
            fo.write('\n')
    fo.close()
        

