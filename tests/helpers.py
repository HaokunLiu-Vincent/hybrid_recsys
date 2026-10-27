"""Small synthetic ratings shared by unit tests; no real dataset access."""
import numpy as np
import pandas as pd

def fixture():
    movies = pd.DataFrame(dict(movieId=np.arange(10,31),title=['Movie '+str(i) for i in range(21)],
                               genres=['Drama','Comedy','Drama|Comedy','(no genres listed)','Action','Drama','Action']*3))
    rows = []
    for user in range(1,9):
        for j in range(6):
            rows.append((user,10+(user+j)%18,4.5 if j < 4 else 2.,100+user*10+j,len(rows)))
    train = pd.DataFrame(rows,columns=['userId','movieId','rating','timestamp','source_row_id'])
    future = pd.DataFrame([(u,11+(u+7)%13,4.5,1000+u,100+u) for u in range(1,9)] +
                          [(99,11,4.5,1010,199),(1,30,5.,1011,200),(2,29,2.,1012,201)],columns=train.columns)
    return train,movies,future

