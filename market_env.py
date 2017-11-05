from random import random
import numpy as np
import math

import gym
from gym import spaces


class MarketEnv(gym.Env):
    """
    株式市場のGym環境
    """
    PENALTY = 1  # 0.999756079

    def __init__(self, dir_path, target_codes, input_codes, start_date, end_date, scope=60, sudden_death=-1.,
                 cumulative_reward=False):
        """
        初期化
        :param dir_path: データのディレクトリへのパス
        :param target_codes: 対象となる銘柄のリスト
        :param input_codes: 対象外の（学習用？）銘柄リスト
        :param start_date: 開始日
        :param end_date: 終了日
        :param scope: 観測対象とする日数
        :param sudden_death: 強制終了フラグ(TODO 条件調べる)
        :param cumulative_reward:
        """
        self.startDate = start_date
        self.endDate = end_date
        self.scope = scope
        self.sudden_death = sudden_death
        self.cumulative_reward = cumulative_reward

        self.inputCodes = []
        self.targetCodes = []
        # 価格情報データマップ。証券コードをキーに価格情報データ（日付をキーに、high, low, close, volumeの変化率のタプルが値）が値
        self.dataMap = {}

        for code in target_codes:
            # for code in (target_codes + input_codes):
            fn = dir_path + "./" + code + ".csv"
            # 価格情報データ。日付をキーに、high, low, close, volumeの変化率のタプルが値
            data = {}
            lastClose = 0
            lastVolume = 0
            try:
                f = open(fn, "r")
                for line in f:
                    if line.strip() != "":
                        dt, openPrice, high, low, close, volume = line.strip().split(",")
                        try:
                            if dt >= start_date:
                                # openは使わず、high, low, close, volumeを取得する
                                high = float(high) if high != "" else float(close)
                                low = float(low) if low != "" else float(close)
                                close = float(close)
                                volume = int(volume)

                                if lastClose > 0 and close > 0 and lastVolume > 0:
                                    # 変化率を取得
                                    close_ = (close - lastClose) / lastClose
                                    high_ = (high - close) / close
                                    low_ = (low - close) / close
                                    volume_ = (volume - lastVolume) / lastVolume

                                    data[dt] = (high_, low_, close_, volume_)

                                lastClose = close
                                lastVolume = volume
                        except Exception as e:
                            print(e, line.strip().split(","))
                f.close()
            except Exception as e:
                print(e)

            if len(data.keys()) > scope:
                self.dataMap[code] = data
                if code in target_codes:
                    self.targetCodes.append(code)
                if code in input_codes:
                    self.inputCodes.append(code)

        self.actions = [
            "LONG",
            "SHORT",
        ]

        self.action_space = spaces.Discrete(len(self.actions))
        # 観測値は変化率なので-1から1の間。
        self.observation_space = spaces.Box(np.ones(scope * (len(input_codes) + 1)) * -1,
                                            np.ones(scope * (len(input_codes) + 1)))

        self.reset()
        self._seed()

    def _step(self, action):
        if self.done:
            return self.state, self.reward, self.done, {}

        self.reward = 0
        if self.actions[action] == "LONG":
            if sum(self.boughts) < 0:
                # ショートがあればクローズ
                for b in self.boughts:
                    # b < -1なら利益、b > -1なら損失
                    self.reward += -(b + 1)
                if self.cumulative_reward:
                    self.reward = self.reward / max(1, len(self.boughts))

                if self.sudden_death * len(self.boughts) > self.reward:
                    self.done = True

                self.boughts = []

            self.boughts.append(1.0)
        elif self.actions[action] == "SHORT":
            if sum(self.boughts) > 0:
                for b in self.boughts:
                    # b > 1なら利益、b < 1なら損失
                    self.reward += b - 1
                if self.cumulative_reward:
                    self.reward = self.reward / max(1, len(self.boughts))

                if self.sudden_death * len(self.boughts) > self.reward:
                    self.done = True

                self.boughts = []

            self.boughts.append(-1.0)
        else:
            pass

        # 終値の変化率を取得
        close_variance = self.target[self.targetDates[self.currentTargetIndex]][2]
        # 終値の変化率を積立て
        self.cum = self.cum * (1 + close_variance)

        for i in range(len(self.boughts)):
            # 倍率の基準。boughtsは売りの時マイナスなので-1にする。
            buy_or_sell = -1 if sum(self.boughts) < 0 else 1
            # 変化率分だけboughtの値を変化させる。損失が出れば1>boughts>-1になる。
            self.boughts[i] = self.boughts[i] * MarketEnv.PENALTY * (1 + close_variance * buy_or_sell)

        self.defineState()
        self.currentTargetIndex += 1
        if self.currentTargetIndex >= len(self.targetDates) or self.endDate <= self.targetDates[
            self.currentTargetIndex]:
            self.done = True

        if self.done:
            for b in self.boughts:
                self.reward += (b * (1 if sum(self.boughts) > 0 else -1)) - 1
            if self.cumulative_reward:
                self.reward = self.reward / max(1, len(self.boughts))

            self.boughts = []

        return self.state, self.reward, self.done, {"dt": self.targetDates[self.currentTargetIndex], "cum": self.cum,
                                                    "code": self.targetCode}

    def _reset(self):
        self.targetCode = self.targetCodes[int(random() * len(self.targetCodes))]
        self.target = self.dataMap[self.targetCode]
        self.targetDates = sorted(self.target.keys())
        self.currentTargetIndex = self.scope
        self.boughts = []
        self.cum = 1.

        self.done = False
        self.reward = 0

        self.defineState()

        return self.state

    def _render(self, mode='human', close=False):
        if close:
            return
        return self.state

    '''
    def _close(self):
        pass

    def _configure(self):
        pass
    '''

    def _seed(self):
        return int(random() * 100)

    def defineState(self):
        """
        stateを作成する
        :return:
        """
        tmpState = []

        # 購買余力。建玉があればその損益によって増減する。
        budget = (sum(self.boughts) / len(self.boughts)) if len(self.boughts) > 0 else 1.
        # ポジションサイズの対数
        size = math.log(max(1., len(self.boughts)), 100)
        # ポジション。買いポジなら1。それ以外は0。
        position = 1. if sum(self.boughts) > 0 else 0.
        # この3つがstate配列の1つ。
        tmpState.append([[budget, size, position]])

        # 対象の終値
        subject = []
        # 対象の出来高
        subjectVolume = []
        for i in range(self.scope):
            try:
                subject.append([self.target[self.targetDates[self.currentTargetIndex - 1 - i]][2]])
                subjectVolume.append([self.target[self.targetDates[self.currentTargetIndex - 1 - i]][3]])
            except Exception as e:
                print(self.targetCode, self.currentTargetIndex, i, len(self.targetDates))
                self.done = True
        tmpState.append([[subject, subjectVolume]])
        tmpState = [np.array(i) for i in tmpState]
        #print(tmpState)
        self.state = tmpState
