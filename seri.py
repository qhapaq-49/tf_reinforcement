# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.contrib import rnn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import random
import copy


"""
seriのルール
各プレイヤーの初期の所持金は10とする。各ラウンドで1,2,3,4の勝利点が順番ランダムで現れる。
各プレイヤーは現れた勝利点をいくらで買うかを宣言。高値をつけたほうが購入できる。
引き分けの場合は購入者無しとして次のラウンドへ向かう（キャリーオーバーはなし）
4ラウンド終了後、所持勝利点が高いほうが勝ち
"""

glob_outdim = 11
# インプットデータ。52 = 11*2*2+4（各プレイヤーの所持金0-10、スコア0-10、目の前にある数字1,2,3,4 and 1,2,3,4の既出）
glob_inpXdim = 52

class seriAgent:
    def __init__(self):
        # 隠れ層は50ニューロンとする
        self.hiddendim = 50
        
        # 出力は購入額0-10の選択確率とする
        self.outdim = glob_outdim
        print("hello seriagent has " + str(glob_inpXdim) + " params")
        
    def design_model(self):
        
        graph_t = tf.Graph()
        with graph_t.as_default():
            # 入力:X、隠れ層:W1,B1,H1,W2、出力:Y
            X  = tf.placeholder(tf.float32, [None, glob_inpXdim])
            W1 = tf.Variable(tf.truncated_normal([glob_inpXdim, self.hiddendim], stddev=0.01), name='W1')
            B1 = tf.Variable(tf.zeros([self.hiddendim]), name='B1')
            H1 = tf.nn.relu(tf.matmul(X, W1) + B1)
            W2 = tf.Variable(tf.random_normal([self.hiddendim, self.outdim], stddev=0.01), name='W2')
            B2 = tf.Variable(tf.zeros([self.outdim]), name='B2')
            Y = tf.nn.softmax(tf.matmul(H1, W2) + B2)
    
            tf.add_to_collection('vars', W1)
            tf.add_to_collection('vars', B1)
            tf.add_to_collection('vars', W2)
            tf.add_to_collection('vars', B2)

            # 学習時の教師データ:t
            t = tf.placeholder(tf.float32, shape=[None, self.outdim])

            # 交差エントロピーの宣言 and 学習機(adam)
            entropy = -tf.reduce_sum(t*tf.log(tf.clip_by_value(Y,1e-10,1.0)))
            learnfunc = tf.train.AdamOptimizer(0.05).minimize(entropy)

            # これらのパラメタを外から弄れるようにする（これ理解正しいの？）
            model = {'X': X, 'Y': Y, 't' : t, 'ent' : entropy, 'learnfunc' : learnfunc}
            self.sess = tf.Session()
            self.saver = tf.train.Saver()
            self.sess.run(tf.global_variables_initializer())

        return model

    def loadnofile(self):
        self.model = self.design_model()
        self.X, self.Y, self.t, self.ent, self.ln  = self.model['X'], self.model['Y'], self.model['t'], self.model['ent'], self.model['learnfunc']
        
        
    def loadfile(self,fname):
        self.loadnofile()
        print("load agent from : " + fname)
        self.saver.restore(self.sess, fname)
        
    def action(self):
        return self.Y
        
    def domove(self, param):
        act = self.action()
        y = self.sess.run(act, feed_dict={self.X: param})
        
        totv = 0.0
        for i in range(self.outdim):
            totv += y[0][i]
            if param[0][i] == 1:
                break
            
        ansv = totv * random.random()
        for i in range(self.outdim):
            ansv -= y[0][i]
            if ansv < 0:
                return i
            if param[0][i] == 1:
                return i
            
    def learnbycsv(self, fname, stepnum,outname):
        # 学習部
        data = pd.read_csv(fname,header=None, dtype=float)
        x = data.iloc[:,0:glob_inpXdim]
        y = data.iloc[:,glob_inpXdim:63]

        for i in range(stepnum):

            self.sess.run(self.ln, feed_dict={self.X : x, self.t : y})
            #print(i)
            if i%20 == 0:
                ent = self.sess.run(self.ent, feed_dict={self.X: x, self.t : y})
                print("epoch " + str(i)+" ,entropy = " + str(ent))

        self.saver.save(self.sess, outname)
# seriを行う部分
class seriGame:
    def __init__(self,fname,fname2):
        self.pos = np.zeros(glob_inpXdim)
        self.pos = np.reshape(self.pos, (-1,glob_inpXdim))
        self.agent = seriAgent() # 複数のモデルを使う場合は此処を変えればいい（はず
        self.agent2 = seriAgent() # 複数のモデルを使う場合は此処を変えればいい（はず
        
        # 1pデータを読み込む
        if fname == "":
            self.agent.loadnofile()
        else:
            self.agent.loadfile(fname)

        # 2pデータも読み込む
        if fname2 == "":
            self.agent2.loadnofile()
        else:
            self.agent2.loadfile(fname2)
            
    def startpos(self):
        # 盤面の初期化
        for i in range(glob_inpXdim):
            self.pos[0][i] = 0
        self.p1 = 0 # 点数を0にする
        self.p2 = 0 # 点数を0にする
        self.m1 = 10 # 所持金を10にする
        self.m2 = 10 # 所持金を10にする

    def setmp2pos(self):
        # 点数と所持金を反映させる
        # どの勝利点が既出かの更新はplayout内部で行うことに注意
        for i in range(44):
            self.pos[0][i] = 0
        self.pos[0][self.m1] = 1
        self.pos[0][11+self.p1] = 1
        self.pos[0][22+self.m2] = 1
        self.pos[0][33+self.p2] = 1

        
    def selectcard(self, pos, ply):
        # まだ出してない勝利点の中から一つをランダムで選択
        app = random.randint(1,4-ply)
        for i in range(4):
            if pos[0][48+i] == 0:
                app -= 1
                if app == 0:
                    return i
        print("no valid card")
        print(pos)
        return -1

    def reversePos(self, posin):
        # 2p用の盤面を生成（変数を反転させる）
        posout = copy.deepcopy(posin)
        for i in range(22):
            posout[0][i] = posin[0][22+i]
            posout[0][i+22] = posin[0][i]
        return posout
            
    def domove_random(self,param):
        # サンドバックとしてランダムムーブを用意しておく
        for i in range(glob_outdim):
            if param[0][i] == 1:
                return random.randint(0,i)
        print("invalid randommove")
        return -1

    def playout(self, gamenum, fname, agent1_random, agent2_random):
        pl1w = 0 # player1の勝利数
        pl2w = 0 # player2の勝利数
        for gloop in range(gamenum):
            self.startpos() # 盤面初期化
            poss = [] # 局面
            plys = [] # 指し手
            for i in range(4):
                self.setmp2pos()
                app = self.selectcard(self.pos,i)
                #print("point = " + str(app+1))
                self.pos[0][44+app] = 1 # ボードのカードとして表示する
                self.pos[0][48+app] = 1 # 既出にする
                
                if agent1_random == False:
                    h1 = self.agent.domove(self.pos)
                else:
                    h1 = self.domove_random(self.pos)
                    
                pose = self.reversePos(self.pos)
                if agent2_random == False:
                    h2 = self.agent2.domove(pose)
                else:
                    h2 = self.domove_random(pose)
                    
                #print("hands = " + str(h1) + "," + str(h2))

                poss.append(copy.deepcopy(self.pos[0]))
                poss.append(copy.deepcopy(pose[0]))
                plys.append(h1)
                plys.append(h2)
                
                if h1 > h2:
                    self.m1 -= h1
                    self.p1 += app+1
                    #print("player 1 bought")
                elif h2 > h1:
                    self.m2 -= h2
                    self.p2 += app+1
                    #print("player 2 bought")
                # ゲームの状況を表示
                #print("player 1 = "+ str(self.m1) + "(money), " + str(self.p1) + "(score)  " + "player 2 = "+ str(self.m2) + "(money), " + str(self.p2) + "(score)" )
                
                for j in range(4):
                    self.pos[0][44+j] = 0
                
            if self.p1 > self.p2 :
                pl1w += 1
                self.generateTeacher(fname, poss, plys, 0)
            elif self.p1 < self.p2 :
                pl2w += 1
                self.generateTeacher(fname, poss, plys, 1)
            if gloop % 1000 == 0:
                print("battle result : " + str(pl1w) + "-" + str(1+gloop-pl1w-pl2w) + "-" + str(pl2w))
                
    def generateTeacher(self, fname, poss, plys, result):
        # ゲームの履歴から教師データを創る
        f = open(fname, 'a')
        for i in range(len(poss)):
            if i%2 != result:
                continue
            outstr = ""
            for j in range(len(poss[i])):
                outstr += str(poss[i][j])+","
            for j in range(self.agent.outdim):
                if j == plys[i]:
                    outstr += "1,"
                else:
                    outstr += "0,"
            f.write(outstr+"\n")
        f.close()

    def learnbycsv(self,fname,stepnum,outname):
        self.agent.learnbycsv(fname,stepnum,outname)

# 自己対戦
playout = seriGame("model4/test_model4","model3/test_model3")
playout.playout(30000,"hoge5",False,False)

# 学習
#sa = seriAgent()
#sa.loadnofile()
#sa.learnbycsv("hoge4",200,"model4/test_model4")
