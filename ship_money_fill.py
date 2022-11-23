# -*- encoding: utf-8 -*-
"""
@Time        : 2022/3/29 09:52
@Author      : SnowBing
@Email       : 834928758@qq.com
@Description : 发货金额补足计算模块
"""
import numpy as np
import pandas as pd
from common import CommonObject
from data_load import mission_info_data_load
from data_load import stock_data_load
from sku_allocation_summary import sku_allocation_summary
from loguru import logger


class ShipMoneyFill(CommonObject):

    def __init__(self):
        super().__init__()
        self.ship_money_fill_df = None
        self.output_columns = [
            'cate_level4_code', 'alloc_cnt_money_fill', 'alloc_reason_money_fill', 'store_sku_inv',
            'sales_pred_d_avg_store_sku_fixed',
            'sales_pred_d_avg_store_sku_fixed_fill_with_avg',
            'sales_pred_d_avg_store_cate4_fixed', 'sku_minimum',
            'sku_norm_cnt', 'warehouse_sku_inv', 'conversion_qty',
            'mainwarehouse', 'notarize_replenishment_amount', 'store_alloc_amt_sum_base', 'store_alloc_amt_sum',
            'combine_price', 'store_sku_avg_day_sale', 'new_arrival'
        ]

    @logger.catch
    def calculate(self) -> pd.DataFrame:
        """
        针对不足门店物流起送金额的门店进行补足操作
        """
        # TODO: 通过引入对象直接传入数据
        if self.ship_money_fill_df is None:
            # 0.数据准备
            # 读取上轮计算的结果
            sku_wrh_realloc_df = sku_allocation_summary.calculate()
            logger.info("start run ShipMoneyFill")
            dim_info = mission_info_data_load.goods_dim()[['combine_price']]
            stock_outstanding_df = stock_data_load.store_sku_stock()[['inv_outstanding']]
            store_cate4_sale_df = mission_info_data_load.store_cate4_sale()[['sales_avg']]  # 门店细类日均销
            # 门店补货金额上下限数据
            store_amount_df = mission_info_data_load.store_amount_info()[
                ['notarize_replenishment_amount', 'amount_section_down',
                 'amount_section_up']]
            logger.info("ShipMoneyFill data load finished!")
            ship_money_fill_df = pd.merge(left=sku_wrh_realloc_df,
                                          right=dim_info,
                                          left_index=True,
                                          right_index=True,
                                          how='left'
                                          )

            ship_money_fill_df['inv_outstanding'] = stock_outstanding_df['inv_outstanding']  # 门店预扣库存
            store_amount_df = store_amount_df.apply(pd.to_numeric, errors='coerce')  # 全部转数字，无法转换则转换成nan
            ship_money_fill_df = pd.merge(left=ship_money_fill_df,
                                          right=store_amount_df,
                                          left_index=True,
                                          right_index=True,
                                          how='left'
                                          )

            #  对数据按照细类日均销从高到低进行排序，若相同，则按照门店代码排序，每个细类内部按照商品代码排序
            ship_money_fill_df = ship_money_fill_df.reset_index().set_index(['store_code', 'cate_level4_code'])
            ship_money_fill_df['sales_avg_store_cate4'] = store_cate4_sale_df['sales_avg']
            ship_money_fill_df.fillna(
                {'combine_price': 0.0, 'inv_outstanding': 0.0, 'sales_avg_store_cate4': 0.0},
                inplace=True
            )
            ship_money_fill_df.reset_index(drop=False, inplace=True)
            ship_money_fill_df.sort_values(
                by=['sales_avg_store_cate4', 'store_code', 'goods_code'],
                ascending=False,
                inplace=True
            )
            ship_money_fill_df.set_index(['store_code', 'goods_code', 'cate_level4_code'], inplace=True)
            day = 0
            ship_money_fill_df['day'] = 0  # 记录补货的天数
            ship_money_fill_df['store_cate4_code'] = ship_money_fill_df.index.get_level_values(
                'store_code').values + ship_money_fill_df.index.get_level_values('cate_level4_code').values
            ship_money_fill_df['alloc_cnt_money_fill'] = ship_money_fill_df['warehouse_alloc_cnt']
            ship_money_fill_df['alloc_reason_money_fill'] = ship_money_fill_df['warehouse_alloc_reason']
            # 记录之前补货金额
            ship_money_fill_df['alloc_amt_money_fill'] = ship_money_fill_df['alloc_cnt_money_fill'].values * \
                                                         ship_money_fill_df['combine_price'].values
            ship_money_fill_df['store_alloc_amt_sum_base'] = ship_money_fill_df.groupby(level=['store_code'])[
                'alloc_amt_money_fill'].transform('sum')
            # 门店预扣金额
            ship_money_fill_df['inv_outstanding_amt'] = ship_money_fill_df['inv_outstanding'].values * \
                                                        ship_money_fill_df['combine_price'].values

            ship_money_fill_df['store_inv_outstanding_amt_sum'] = ship_money_fill_df.groupby(level=['store_code'])[
                'inv_outstanding_amt'].transform('sum')

            while True:
                # 计算仓库分配完毕后可用库存
                ship_money_fill_df['warehouse_sku_inv_res'] = ship_money_fill_df['warehouse_sku_inv'].values - \
                                                              ship_money_fill_df['alloc_cnt_money_fill'].values

                # 一、 起送金额缺口计算
                ship_money_fill_df['alloc_amt_money_fill'] = ship_money_fill_df['alloc_cnt_money_fill'].values * \
                                                             ship_money_fill_df['combine_price']
                ship_money_fill_df['store_alloc_amt_sum'] = ship_money_fill_df.groupby(level=['store_code'])[
                    'alloc_amt_money_fill'].transform('sum')

                # 当前下单金额+预扣金额
                ship_money_fill_df['store_alloc_outstanding_amt_sum'] = \
                    ship_money_fill_df['store_alloc_amt_sum'].values + \
                    ship_money_fill_df['store_inv_outstanding_amt_sum'].values
                # 门店起送金额缺口 = max(仓库起送金额上限 - 门店预扣库存金额 - 门店建议下单金额, 门店目标下单金额 - 门店建议下单金额)
                ship_money_fill_df['store_alloc_amt_res'] = np.max(
                    [ship_money_fill_df['amount_section_up'].fillna(0.0).values -
                     ship_money_fill_df['store_alloc_outstanding_amt_sum'].values,
                     ship_money_fill_df['notarize_replenishment_amount'].fillna(0.0).values -
                     ship_money_fill_df['store_alloc_amt_sum'].values
                     ],
                    axis=0
                )
                # 目标金额为空，且预扣+建议不在区间范围内
                ship_money_fill_df['store_alloc_amt_res'] = np.where(
                    (ship_money_fill_df['notarize_replenishment_amount'].isnull()) &
                    (ship_money_fill_df['store_alloc_outstanding_amt_sum'] < ship_money_fill_df['amount_section_down']),
                    0.0,
                    ship_money_fill_df['store_alloc_amt_res']
                )
                # 如果所有门店的金额缺口都小于0，则停止循环，否则进入金额补足模块
                if (len(ship_money_fill_df[ship_money_fill_df['store_alloc_amt_res'] > 0.0]) == 0):
                    logger.info("all store satisfy for ship money")
                    break
                else:
                    # 最高按照销量增加的天数
                    if day < 30:
                        logger.info(f"ship money fill day: {day + 1}")
                        # 二、金额补足模块
                        # 1.筛选补货子集：
                        #   a.当前仓库仍有剩余库存；
                        #   b.当前金额缺口>0.0；
                        #   c.销量预测大于0；
                        #   d.剔除掉细类80040101 装材购物袋；
                        #   e.补足操作新增上限不得超过100
                        #   f.前期分配环节库存不足分配未0的情况
                        #   g.细类定标不为0
                        # 每次新增一天
                        ship_money_fill_df['add1day_cnt'] = np.where(
                            (ship_money_fill_df['warehouse_sku_inv_res'].values > 0) &
                            (ship_money_fill_df['store_alloc_amt_res'].values > 0.0) &
                            (ship_money_fill_df['sales_pred_d_avg_store_sku_fixed_fill_with_avg'] > 0.0) &
                            (ship_money_fill_df.index.get_level_values('cate_level4_code') != '80040101') &
                            (ship_money_fill_df['alloc_cnt_money_fill'] - ship_money_fill_df[
                                'warehouse_alloc_cnt'] < 100.0) &
                            (~ship_money_fill_df['wrh_alloc_no_fit'].values) &
                            (ship_money_fill_df['sku_qty'] > 0) &
                            (ship_money_fill_df['pcs_qty'] > 0),
                            ship_money_fill_df['sales_pred_d_avg_store_sku_fixed_fill_with_avg'],
                            0.0
                        )
                        # 此处分配要跟前一天分配结果进行前后对比，若两者之差差一个规格，那么就进行补货；否则不补
                        # 对当前新增结果进行规格处理，对结果进行2舍3入取整，之后乘以规格
                        ship_money_fill_df['alloc_cnt_tmp'] = np.multiply(
                            np.round(
                                (ship_money_fill_df['add1day_cnt'].values * (ship_money_fill_df['day'].values + 1)) /
                                ship_money_fill_df['conversion_qty'].values
                                + 0.2 + 1e-8
                            ),
                            ship_money_fill_df['conversion_qty'].values
                        ) - np.multiply(
                            np.round(
                                (ship_money_fill_df['add1day_cnt'].values * ship_money_fill_df['day'].values)
                                / ship_money_fill_df['conversion_qty'].values
                                + 0.2 + 1e-8
                            ),
                            ship_money_fill_df['conversion_qty'].values
                        )
                        # 对补货后的周转天数进行控制，不得超过35+30天，避免部分单品周转过高
                        ship_money_fill_df['alloc_cnt_tmp'] = np.where(
                            np.sum(ship_money_fill_df[['store_sku_inv', 'alloc_cnt_money_fill', 'alloc_cnt_tmp']],
                                   axis=1) <
                            ship_money_fill_df['store_sku_avg_day_sale'].values * 65,
                            ship_money_fill_df['alloc_cnt_tmp'].values,
                            0.0
                        )
                        # 计算当前仓库单品库存剩余量，过滤掉库存不足的情况
                        ship_money_fill_df['alloc_cnt_tmp_cumsum'] = ship_money_fill_df.groupby(
                            level=['goods_code'])['alloc_cnt_tmp'].transform('cumsum')
                        # 计算仓库库存分配完后库存量
                        ship_money_fill_df['alloc_cnt_tmp_cumsum_res'] = \
                            ship_money_fill_df['warehouse_sku_inv_res'].values - \
                            ship_money_fill_df['alloc_cnt_tmp_cumsum'].values
                        # 计算每步分配前一步剩余库存量
                        ship_money_fill_df['alloc_cnt_tmp_cumsum_res_last'] = \
                            ship_money_fill_df['alloc_cnt_tmp_cumsum_res'].values - \
                            ship_money_fill_df['alloc_cnt_tmp'].values

                        # 计算最终的分配量
                        ship_money_fill_df['alloc_cnt_tmp'] = np.min(
                            ship_money_fill_df[['alloc_cnt_tmp', f'alloc_cnt_tmp_cumsum_res_last']],
                            axis=1
                        )

                        # 将每步分配前已经库存小于0的数据全部置为0
                        ship_money_fill_df.loc[
                            ship_money_fill_df['alloc_cnt_tmp_cumsum_res_last'] < 0.0, 'alloc_cnt_tmp'] = 0.0

                        # 再次对分配结果进行规格处理，需要注意的是此处采用下舍取整，保证结果一定不会超过可用库存总量，得到最终本次的分配量
                        ship_money_fill_df['alloc_cnt_tmp'] = np.multiply(
                            np.floor(
                                np.divide(ship_money_fill_df['alloc_cnt_tmp'].values,
                                          ship_money_fill_df['conversion_qty'].values)
                            ),
                            ship_money_fill_df['conversion_qty']
                        )

                        ship_money_fill_df['alloc_amt_tmp'] = ship_money_fill_df['alloc_cnt_tmp'].values * \
                                                              ship_money_fill_df['combine_price'].values
                        # 按照细类累加补货量，如果达到缺口，则将后续补货量置为0
                        ship_money_fill_df['alloc_amt_store_goods_cumsum'] = ship_money_fill_df.groupby(
                            level=['store_code'])['alloc_amt_tmp'].cumsum()
                        ship_money_fill_df['alloc_amt_store_cate4_sum'] = ship_money_fill_df.groupby(
                            level=['store_code', 'cate_level4_code'])['alloc_amt_tmp'].transform('sum')

                        ship_money_fill_df['alloc_amt_store_cate4_cumsum'] = ship_money_fill_df.store_cate4_code.map(
                            ship_money_fill_df.groupby('store_cate4_code')['alloc_amt_store_goods_cumsum'].last()
                        )
                        # 计算当前细类分配上剩余缺口，用于判断是否停止加货
                        ship_money_fill_df['store_alloc_amt_res_last'] = \
                            ship_money_fill_df['store_alloc_amt_res'].values - \
                            ship_money_fill_df['alloc_amt_store_cate4_cumsum'].values + \
                            ship_money_fill_df['alloc_amt_store_cate4_sum'].values

                        # 将达到金额缺口的补货量置为0
                        ship_money_fill_df.loc[
                            ship_money_fill_df['store_alloc_amt_res_last'] <= 0.0, 'alloc_cnt_tmp'] = 0.0
                        # 将当前补货量累加到重分配量上
                        ship_money_fill_df['alloc_cnt_money_fill'] += ship_money_fill_df['alloc_cnt_tmp']

                        # 若当前步骤该商品补货了，则将其day加1
                        ship_money_fill_df['day'] += np.where(
                            ship_money_fill_df['add1day_cnt'] > 0.0,
                            1,
                            0.0
                        )
                        # 记录已经补货的迭代天数
                        day += 1
                    else:
                        # 已加到30天，但仍有部门门店不满足起送金额，则剩余部分不用添加
                        logger.warning(
                            "ship money fill task:still have unsatisfied records and fill day have add to 30 day!"
                        )
                        break
            ship_money_fill_df.reset_index('cate_level4_code', drop=False, inplace=True)
            ship_money_fill_df['alloc_reason_money_fill'] += \
                "_金额补足:补前金额" + \
                np.round(ship_money_fill_df['store_alloc_amt_sum_base'] + 1e-8).astype('int').astype('str') + \
                ",预扣金额" + np.round(ship_money_fill_df['store_inv_outstanding_amt_sum'] + 1e-8).astype('int').astype(
                    'str') + \
                np.where(
                    # 目标金额为空的情况下，判断是因为空还是不满足区间导致无法填充
                    ship_money_fill_df['notarize_replenishment_amount'].isnull(),
                    ",目标金额空" +
                    np.select(
                        condlist=[
                            ship_money_fill_df['amount_section_down'].isnull(),
                            ship_money_fill_df['store_alloc_outstanding_amt_sum'] < ship_money_fill_df[
                                'amount_section_down'],
                            ship_money_fill_df['store_alloc_outstanding_amt_sum'] > ship_money_fill_df[
                                'amount_section_up']],
                        choicelist=[
                            ",上下限空",
                            ",小于下限" + np.round(ship_money_fill_df['amount_section_down'] + 1e-8).fillna(0).astype(
                                'int').astype('str'),
                            ",大于上限" + np.round(ship_money_fill_df['amount_section_up'] + 1e-8).fillna(0).astype(
                                'int').astype('str'),
                        ],
                        default=""
                    ),
                    # 目标金额非空，则记录
                    ",目标金额" + np.round(ship_money_fill_df['notarize_replenishment_amount'] + 1e-8).fillna(0).astype(
                        'int').astype('str') +
                    np.where(
                        ship_money_fill_df['amount_section_down'].isnull(),
                        ",上下限空",
                        ",金额区间(" +
                        np.round(ship_money_fill_df['amount_section_down'] + 1e-8).fillna(0).astype('int').astype(
                            'str') + "," +
                        np.round(ship_money_fill_df['amount_section_up'] + 1e-8).fillna(0).astype('int').astype(
                            'str') + ")"
                    )
                ) + \
                np.where(
                    (ship_money_fill_df['day'].values > 0) &
                    (ship_money_fill_df['alloc_cnt_money_fill'] > ship_money_fill_df['warehouse_alloc_cnt']),
                    ",扩增" + ship_money_fill_df['day'].fillna(0).astype('int').astype('str') + '天',
                    ",本商品未扩增"
                ) + \
                ",补后金额" + np.round(ship_money_fill_df['store_alloc_amt_sum'] + 1e-8).astype('int').astype('str') + \
                ',补后数量' + np.round(ship_money_fill_df['alloc_cnt_money_fill'] + 1e-8).astype('int').astype('str')
            self.ship_money_fill_df = ship_money_fill_df[self.output_columns]
            # 完成金额补足操作，返回最终结果
            self.check_amount(self.ship_money_fill_df, len(sku_wrh_realloc_df))  # 数据量检测
            self.check_negative(self.ship_money_fill_df, 'alloc_cnt_money_fill')  # 非负检测
            self.check_zero(self.ship_money_fill_df, 'alloc_cnt_money_fill')  # 0值占比检测
            logger.info("ShipMoneyFill finished!")
        return self.ship_money_fill_df


ship_money_fill = ShipMoneyFill()
