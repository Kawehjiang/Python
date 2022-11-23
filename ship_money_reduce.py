# -*- encoding: utf-8 -*-
"""
@Time        : 2022/7/11 09:21
@Author      : SnowBing
@Email       : 834928758@qq.com
@Description : 发货金额降低计算模块
"""
import numpy as np
import pandas as pd
from common import CommonObject
from ship_money_fill import ship_money_fill
from loguru import logger


class ShipMoneyReduce(CommonObject):
    """在当前补货金额大于建议补货金额的105%时，执行此步骤"""

    def __init__(self):
        self.ship_money_reduce_df = None
        self.output_columns = [
            'cate_level4_code', 'alloc_cnt_money_reduce', 'alloc_reason_money_reduce', 'store_sku_inv',
            'sales_pred_d_avg_store_sku_fixed',
            'sales_pred_d_avg_store_sku_fixed_fill_with_avg',
            'sales_pred_d_avg_store_cate4_fixed', 'sku_minimum',
            'sku_norm_cnt', 'warehouse_sku_inv', 'conversion_qty',
            'mainwarehouse'
        ]

    @logger.catch
    def calculate(self) -> pd.DataFrame:
        if self.ship_money_reduce_df is None:
            # 计算前面的步骤
            ship_money_reduce_df = ship_money_fill.calculate().copy()
            ship_money_reduce_df['alloc_cnt_money_reduce'] = ship_money_reduce_df['alloc_cnt_money_fill']
            ship_money_reduce_df['store_alloc_amt_sum_reduce'] = ship_money_reduce_df['store_alloc_amt_sum']
            ship_money_reduce_df['notarize_replenishment_amount_target'] = \
                ship_money_reduce_df['notarize_replenishment_amount'].values * 1.05
            logger.info("start run ShipMoneyReduce")
            times = 0
            while times < 30:
                # 判断是否存在需要减补金额的门店，必须同时满足以下3个条件
                # 1）建议金额大于0
                # 2) 补货金额大于目标金额的105%
                # 3) 金额补足模块未涉及到
                ship_money_reduce_df['whether_reduce'] = False  # 默认为否
                ship_money_reduce_df.loc[
                    (ship_money_reduce_df['notarize_replenishment_amount_target'].values > 0.0) &
                    (ship_money_reduce_df['store_alloc_amt_sum_reduce'].values >
                     ship_money_reduce_df['notarize_replenishment_amount_target']) &
                    (ship_money_reduce_df['store_alloc_amt_sum'] == ship_money_reduce_df['store_alloc_amt_sum_base']),
                    'whether_reduce'
                ] = True
                # 如果存在任意一家门店要进行减补步骤，那么则进入循环
                if np.any(ship_money_reduce_df['whether_reduce'].values):
                    # 计算门店超出金额
                    ship_money_reduce_df['alloc_amt_money_diff'] = np.where(
                        ship_money_reduce_df['whether_reduce'],
                        ship_money_reduce_df['store_alloc_amt_sum_reduce'] - ship_money_reduce_df[
                            'notarize_replenishment_amount_target'],
                        0
                    )
                    # 计算单品可售天数
                    ship_money_reduce_df['store_sku_avail_sale_day'] = np.where(
                        ship_money_reduce_df['store_sku_avg_day_sale'].values <= 0.0,
                        1000,
                        (ship_money_reduce_df['store_sku_inv'] + ship_money_reduce_df['alloc_cnt_money_reduce']) / \
                        ship_money_reduce_df['store_sku_avg_day_sale'].values
                    )
                    # 倒序排序
                    ship_money_reduce_df.sort_values(by=['store_sku_avail_sale_day'], ascending=False, inplace=True)
                    # 针对每个有补货量且需要减补的门店，每个减少一个规格,其中过滤以下商品
                    # 1) 不处理新品
                    # 2) 不处理塑料袋细类
                    # 暂时先下线过滤条件
                    ship_money_reduce_df['reduce_cnt'] = np.where(
                        (ship_money_reduce_df['whether_reduce']) &
                        # (ship_money_reduce_df['new_arrival'] != 1) &
                        # (ship_money_reduce_df['cate_level4_code'] != '80040101') &
                        (ship_money_reduce_df['alloc_cnt_money_reduce'] > 0.0),
                        ship_money_reduce_df['conversion_qty'],
                        0.0
                    )
                    ship_money_reduce_df['reduce_amt'] = ship_money_reduce_df['reduce_cnt'] * ship_money_reduce_df[
                        'combine_price']
                    # 计算累减金额
                    ship_money_reduce_df['reduce_amt_sum'] = ship_money_reduce_df.groupby(level=['store_code'])[
                        'reduce_amt'].cumsum()
                    # 计算删除当前sku前累计删减金额
                    ship_money_reduce_df['reduce_amt_sum_last'] = ship_money_reduce_df['reduce_amt_sum'].values - \
                                                                  ship_money_reduce_df['reduce_amt'].values
                    # 若减少当前sku一个规格还未达到指定金额，那么无论如何，继续减去当前sku一个规格
                    ship_money_reduce_df.loc[
                        ship_money_reduce_df['reduce_amt_sum_last'] > ship_money_reduce_df['alloc_amt_money_diff'],
                        ['reduce_cnt', 'reduce_amt']
                    ] = 0.0
                    # 记录本轮要减去的补货金额
                    ship_money_reduce_df['reduce_amt_store_sum'] = ship_money_reduce_df.groupby('store_code')[
                        'reduce_amt'].transform('sum')
                    ship_money_reduce_df['store_alloc_amt_sum_reduce'] -= ship_money_reduce_df['reduce_amt_store_sum']
                    # 更新补货量
                    ship_money_reduce_df['alloc_cnt_money_reduce'] -= ship_money_reduce_df['reduce_cnt']
                    times += 1
                    logger.info(f"ShipMoneyReduce calculate times :{times}")
                    continue
                else:
                    break
            # 更新补货量和补货原因
            # 计算总共减少了几个规格
            ship_money_reduce_df['conversion_num'] = np.round(
                (ship_money_reduce_df['alloc_cnt_money_fill'].values -
                 ship_money_reduce_df['alloc_cnt_money_reduce'].values + 1e-8) / \
                ship_money_reduce_df['conversion_qty'].values
            )
            ship_money_reduce_df['alloc_reason_money_reduce'] = \
                ship_money_reduce_df['alloc_reason_money_fill'] + "_金额减少:" + \
                np.where(
                    ship_money_reduce_df['store_alloc_amt_sum'] > ship_money_reduce_df['store_alloc_amt_sum_reduce'],
                    "减后门店金额" + np.round(ship_money_reduce_df['store_alloc_amt_sum_reduce'] + 1e-8).astype(int).astype(
                        'str'),
                    ""
                ) + \
                np.where(
                    ship_money_reduce_df['conversion_num'] > 0.0,
                    ",减少" + ship_money_reduce_df['conversion_num'].astype(int).astype('str') + "个规格",
                    "未减少"
                ) + ",最终:" + np.round(ship_money_reduce_df['alloc_cnt_money_reduce'] + 1e-8).astype('int').astype('str')
            self.ship_money_reduce_df = ship_money_reduce_df[self.output_columns]
            self.check_amount(self.ship_money_reduce_df, len(ship_money_fill.calculate()))
            self.check_negative(self.ship_money_reduce_df, 'alloc_cnt_money_reduce')
            self.check_zero(self.ship_money_reduce_df, 'alloc_cnt_money_reduce')
            logger.info("ShipMoneyReduce finished!")

        return self.ship_money_reduce_df


ship_money_reduce = ShipMoneyReduce()
