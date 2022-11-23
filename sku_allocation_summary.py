# -*- encoding: utf-8 -*-
"""
@Time        : 2022/6/10 14:38
@Author      : SnowBing
@Email       : 834928758@qq.com
@Description : 两个分配环节融合在一起
"""
import numpy as np
import pandas as pd
from common import CommonObject
from sku_init_allocation import sku_init_allocation
from sku_warehouse_reallocation import sku_warehouse_reallocation
from loguru import logger


class SkuAllocationSummary(CommonObject):

    def __init__(self):
        super().__init__()
        #  sku分配结果
        self.sku_alloc_df = None
        self.output_columns = ['warehouse_alloc_cnt', 'warehouse_alloc_reason',
                               'store_sku_inv', 'sales_pred_d_avg_store_sku_fixed',
                               'sales_pred_d_avg_store_sku_fixed_fill_with_avg',
                               'sales_pred_d_avg_store_cate4_fixed',
                               'cate_level4_code', 'sku_minimum', 'sku_norm_cnt', 'warehouse_sku_inv', 'conversion_qty',
                               'mainwarehouse', 'store_sku_avg_day_sale', 'wrh_alloc_no_fit', 'new_arrival',
                               'sku_qty', 'pcs_qty']

    def calculate(self) -> pd.DataFrame:
        if self.sku_alloc_df is None:
            # 数据准备
            input_data_df = sku_init_allocation.data_prepare()
            input_data_df_len = len(input_data_df)
            logger.info("SkuAllocation data load finished!")
            round_alloc = True  # 判断是否进入循环分配
            if round_alloc:
                alloc_times = 0
                logger.warning("You choose  allocation task multiple times,this will cost more time!")
                while alloc_times < 2:
                    # 循环分配环节：当前循环最大次数为2
                    # 初始分配，先清空上次循环数据
                    logger.info(f"start run allocation task : {alloc_times + 1} time")
                    if 'warehouse_alloc_cnt' in input_data_df.columns:
                        del input_data_df['warehouse_alloc_cnt'], input_data_df['warehouse_alloc_reason'], \
                            input_data_df['alloc_cnt']
                    input_data_df = sku_init_allocation.sku_init_allocation_cal(input_data_df)
                    # 重分配
                    input_data_df = sku_warehouse_reallocation.sku_warehouse_reallocation_cal(input_data_df)
                    # 将当前分配量假定为门店库存，更新数据，重新进入分配环节
                    input_data_df = input_data_df.reset_index().set_index(
                        ['store_code', 'goods_code', 'cate_level4_code'])
                    # 1.门店sku可用库存
                    input_data_df['store_sku_inv'] += input_data_df['warehouse_alloc_cnt']
                    # 2.细类补货空间
                    input_data_df['wrh_alloc_cnt_store_cate4_sum'] = input_data_df.groupby(
                        level=['store_code', 'cate_level4_code'])['warehouse_alloc_cnt'].transform('sum')
                    input_data_df['cate4_replenishment_space'] -= input_data_df['wrh_alloc_cnt_store_cate4_sum']
                    # 3.sku基础补货量及上限
                    input_data_df['sku_init_replenishment'] -= input_data_df['warehouse_alloc_cnt']
                    input_data_df['sku_init_replenishment_upper'] -= input_data_df['warehouse_alloc_cnt']
                    # 4.更新仓库库存
                    input_data_df['wrh_alloc_cnt_sku_sum'] = input_data_df.groupby(level=['goods_code'])[
                        'warehouse_alloc_cnt'].transform('sum')
                    input_data_df['warehouse_sku_inv'] -= input_data_df['wrh_alloc_cnt_sku_sum']
                    # 将小于0的元素置为0
                    non_negative_columns = ['cate4_replenishment_space', 'sku_init_replenishment',
                                            'sku_init_replenishment_upper',
                                            'wrh_alloc_cnt_sku_sum', 'warehouse_sku_inv']
                    input_data_df[non_negative_columns] = input_data_df[non_negative_columns].clip(0)
                    # 记录下首次分配的结果
                    if alloc_times == 0:
                        alloc_reason_first_time = input_data_df[['warehouse_alloc_reason']]
                    alloc_times += 1
                # 计算初始分配最终结果=门店库存变化值
                input_data_df['warehouse_alloc_cnt'] = input_data_df['store_sku_inv'] - input_data_df[
                    'store_sku_inv_base']
                # 将门店库存重置为原始输入
                input_data_df['store_sku_inv'] = input_data_df['store_sku_inv_base']
                # 将仓库库存重置为原始输入
                input_data_df['warehouse_sku_inv'] = input_data_df['warehouse_sku_inv_base']
                # 重新整理补货原因
                input_data_df['warehouse_alloc_reason'] = \
                    alloc_reason_first_time['warehouse_alloc_reason'] + \
                    f"_累计分配{alloc_times}轮结果:" \
                    + np.round(input_data_df['warehouse_alloc_cnt'] + 1e-8).astype('int').astype('str')
                input_data_df.reset_index('cate_level4_code', drop=False, inplace=True)
            else:
                # 初始分配
                input_data_df = sku_init_allocation.sku_init_allocation_cal(input_data_df)
                # 重分配
                input_data_df = sku_warehouse_reallocation.sku_warehouse_reallocation_cal(input_data_df)

            # 记录下最终一轮，因为仓库库存不足，导致的未分配到的记录：需求>0,但未分配的记录
            input_data_df['wrh_alloc_no_fit'] = np.where(
                (input_data_df['alloc_cnt'] > 0.0) &
                (input_data_df['warehouse_alloc_cnt'] == 0.0),
                True,
                False
            )

            # 整理最终输出结果
            self.sku_alloc_df = input_data_df[self.output_columns]
            self.check_amount(self.sku_alloc_df, input_data_df_len)  # 数据量检测
            self.check_negative(self.sku_alloc_df, 'warehouse_alloc_cnt')  # 非负检测
            self.check_zero(self.sku_alloc_df, 'warehouse_alloc_cnt')  # 0值占比检测

        logger.info("SkuAllocation finished!")
        return self.sku_alloc_df


sku_allocation_summary = SkuAllocationSummary()
