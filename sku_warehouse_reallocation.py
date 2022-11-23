# -*- encoding: utf-8 -*-
"""
@Time        : 2022/2/23 3:44 PM
@Author      : miniso
@Description : 仓库重分配计算模块
"""
import pandas as pd
import numpy as np
from loguru import logger
from common import CommonObject
from sku_init_allocation import sku_init_allocation


class SkuWarehouseReallocation(CommonObject):

    def __init__(self):
        super().__init__()
        self.sku_warehouse_reallocation_df = None
        self.output_columns = ['alloc_cnt', 'warehouse_alloc_cnt', 'warehouse_alloc_reason',
                               'store_sku_inv', 'sales_pred_d_avg_store_sku_fixed',
                               'sales_pred_d_avg_store_sku_fixed_fill_with_avg',
                               'sales_pred_d_avg_store_cate4_fixed',
                               'cate_level4_code', 'sku_minimum', 'sku_norm_cnt', 'warehouse_sku_inv',
                               'conversion_qty', 'cate4_need', 'cate4_no_need',
                               'cate4_replenishment_space', 'cate_level1_code', 'mainwarehouse',
                               'new_arrival', 'new_store', 'series_sku_tag',
                               'sku_init_replenishment', 'sku_init_replenishment_upper', 'sku_qty', 'pcs_qty',
                               'store_cate1_avg_day_sale', 'store_sku_avg_day_sale',
                               'store_sku_display_minimum', 'store_sku_priority',
                               'store_sku_week_fix_avg_day_sale',
                               'top_sales', 'whether_abolish', 'wrh_sku_display_minimum',
                               'store_sku_inv_base', 'warehouse_sku_inv_base', 'sku_init_replenishment_base',
                               'sku_init_replenishment_upper_base',
                               'cate4_replenishment_space_base', 'sku_minimum_cate4_sum', 'sale_limit', 'sale_day_limit'
                               ]

    @logger.catch
    def calculate(self) -> pd.DataFrame:
        if self.sku_warehouse_reallocation_df is None:
            # 初始分配计算
            sku_init_alloc_df = sku_init_allocation.calculate()
            logger.info("start run SkuWarehouseReallocation")
            self.sku_warehouse_reallocation_df = self.sku_warehouse_reallocation_cal(sku_init_alloc_df)
            self.check_amount(self.sku_warehouse_reallocation_df, len(sku_init_alloc_df))
            self.check_zero(self.sku_warehouse_reallocation_df, 'warehouse_alloc_cnt')
            self.check_negative(self.sku_warehouse_reallocation_df, 'warehouse_alloc_cnt')
            logger.info("SkuWarehouseReallocation finished!")

        # 输出最终结果
        return self.sku_warehouse_reallocation_df

    @logger.catch
    def sku_warehouse_reallocation_cal(self, input_data_df: pd.DataFrame) -> pd.DataFrame:
        """
        重分配计算
        """
        sku_init_alloc_df = input_data_df.copy(deep=True)
        # 定义一个下文常用的空数据集
        _TMP_df = pd.DataFrame(columns=['store_code', 'goods_code'] + self.output_columns)
        _TMP_df.set_index(["store_code", "goods_code"], inplace=True)
        sku_init_alloc_df['alloc_cnt'].fillna(0.0, inplace=True)
        sku_init_alloc_df["alloc_cnt_sum"] = sku_init_alloc_df.groupby(level=['goods_code'])["alloc_cnt"].transform(
            "sum")
        sku_wrh_realloc_df = sku_init_alloc_df.copy(deep=True)
        sku_wrh_realloc_df['alloc_reason'] += \
            "_重分配:仓库库存" + np.round(sku_wrh_realloc_df['warehouse_sku_inv'] + 1e-8).astype(int).astype('str')
        sku_wrh_realloc_df_stock_zero = sku_wrh_realloc_df[sku_wrh_realloc_df['warehouse_sku_inv'] <= 0]

        if len(sku_wrh_realloc_df_stock_zero) == 0:
            sku_wrh_realloc_df_stock_zero = _TMP_df.copy(deep=True)
        else:
            sku_wrh_realloc_df_stock_zero['alloc_cnt'] = 0.0
            sku_wrh_realloc_df_stock_zero['alloc_reason'] += ',仓库无库存不分配'
            sku_wrh_realloc_df_stock_zero[['warehouse_alloc_cnt', 'warehouse_alloc_reason']] = \
                sku_wrh_realloc_df_stock_zero[['alloc_cnt', 'alloc_reason']]
            # sku_wrh_realloc_df_stock_zero.rename(
            #     columns={'alloc_cnt': 'warehouse_alloc_cnt', 'alloc_reason': 'warehouse_alloc_reason'},
            #     inplace=True
            # )
            sku_wrh_realloc_df_stock_zero = sku_wrh_realloc_df_stock_zero[self.output_columns]
        # b.补货量小于仓库库存全量分配
        sku_wrh_realloc_df_satisfy = sku_wrh_realloc_df[
            (sku_wrh_realloc_df['alloc_cnt_sum'] <= sku_wrh_realloc_df['warehouse_sku_inv']) &
            (sku_wrh_realloc_df['warehouse_sku_inv'] > 0)
            ]
        if len(sku_wrh_realloc_df_satisfy) == 0:
            sku_wrh_realloc_df_satisfy = _TMP_df.copy(deep=True)
        else:
            sku_wrh_realloc_df_satisfy['alloc_reason'] += \
                ",库存充足全量分配" + sku_wrh_realloc_df_satisfy['alloc_cnt'].astype(int).astype('str')
            sku_wrh_realloc_df_satisfy[['warehouse_alloc_cnt', 'warehouse_alloc_reason']] = \
                sku_wrh_realloc_df_satisfy[['alloc_cnt', 'alloc_reason']]
            sku_wrh_realloc_df_satisfy = sku_wrh_realloc_df_satisfy[self.output_columns]
        # 拼接直接输出不用分配的任务单
        sku_wrh_realloc_df_all_satisfy = pd.concat([sku_wrh_realloc_df_stock_zero, sku_wrh_realloc_df_satisfy],
                                                   ignore_index=False
                                                   )

        # c.抽取库存不足分配的任务单
        sku_wrh_realloc_df_not_satisfy = sku_wrh_realloc_df[
            (sku_wrh_realloc_df['alloc_cnt_sum'] > sku_wrh_realloc_df['warehouse_sku_inv']) &
            (sku_wrh_realloc_df['warehouse_sku_inv'] > 0)
            ]

        # 若所有商品需求量均满足，则直接返回结果
        if len(sku_wrh_realloc_df_not_satisfy) == 0:
            return sku_wrh_realloc_df_all_satisfy

        # 二、根据优先级对仓库库存进行重分配
        # 基于是否新品将任务进行切分
        sku_wrh_realloc_df_not_satisfy_new_sku = sku_wrh_realloc_df_not_satisfy[
            sku_wrh_realloc_df_not_satisfy['new_arrival'] == 1]
        sku_wrh_realloc_df_not_satisfy_old_sku = sku_wrh_realloc_df_not_satisfy[
            sku_wrh_realloc_df_not_satisfy['new_arrival'] != 1]

        # 2.1 对新品进行重分配
        # 先判断是否存在需要分配的新品
        if len(sku_wrh_realloc_df_not_satisfy_new_sku) > 0:
            # 2.1.1 首先对新品需求量是否超过仓库可用量进行数据切分
            sku_wrh_realloc_df_not_satisfy_new_sku['alloc_cnt_new_sku_sum'] = \
                sku_wrh_realloc_df_not_satisfy_new_sku.groupby(level="goods_code")["alloc_cnt"].transform("sum")
            # 过滤库存满足的商品
            sku_wrh_realloc_df_not_satisfy_new_sku_satisfy = sku_wrh_realloc_df_not_satisfy_new_sku[
                sku_wrh_realloc_df_not_satisfy_new_sku[
                    "alloc_cnt_new_sku_sum"] <= \
                sku_wrh_realloc_df_not_satisfy_new_sku[
                    "warehouse_sku_inv"]
                ]

            if len(sku_wrh_realloc_df_not_satisfy_new_sku_satisfy) > 0:
                sku_wrh_realloc_df_not_satisfy_new_sku_satisfy["warehouse_alloc_cnt"] = \
                    sku_wrh_realloc_df_not_satisfy_new_sku_satisfy['alloc_cnt']
                sku_wrh_realloc_df_not_satisfy_new_sku_satisfy['warehouse_alloc_reason'] = \
                    sku_wrh_realloc_df_not_satisfy_new_sku_satisfy["alloc_reason"] + \
                    ",新品库存充足全量分配" + sku_wrh_realloc_df_not_satisfy_new_sku_satisfy['warehouse_alloc_cnt'].astype('str')
                sku_wrh_realloc_df_not_satisfy_new_sku_satisfy = sku_wrh_realloc_df_not_satisfy_new_sku_satisfy[
                    self.output_columns]
            else:
                sku_wrh_realloc_df_not_satisfy_new_sku_satisfy = _TMP_df.copy(deep=True)

            # 针对不满足库存的新品按照门店大类优先级分配
            sku_wrh_realloc_df_not_satisfy_new_sku_not_satisfy = sku_wrh_realloc_df_not_satisfy_new_sku[
                sku_wrh_realloc_df_not_satisfy_new_sku[
                    "alloc_cnt_new_sku_sum"] > \
                sku_wrh_realloc_df_not_satisfy_new_sku[
                    "warehouse_sku_inv"]
                ]

            if len(sku_wrh_realloc_df_not_satisfy_new_sku_not_satisfy) > 0:
                # 大类门店优先级计算
                # 仅筛选需要的门店*大类进行优先级计算
                # 1. 大类门店优先级:首先将新店前置，其次按照门店大类日均销对各个商品，从大到小排序，相同时按照门店代码排序
                # index仅保留goods_code
                sku_wrh_realloc_df_not_satisfy_new_sku_not_satisfy.reset_index(
                    level='store_code',
                    drop=False,
                    inplace=True
                )
                # 计算单品在不同门店的大类优先级
                sku_wrh_realloc_df_not_satisfy_new_sku_not_satisfy.sort_values(
                    by=['new_store', 'store_cate1_avg_day_sale', 'store_code'],
                    ascending=False, inplace=True
                )
                sku_wrh_realloc_df_not_satisfy_new_sku_not_satisfy['store_cate1_priority'] = \
                    np.arange(1, len(sku_wrh_realloc_df_not_satisfy_new_sku_not_satisfy) + 1)
                # 索引处理
                sku_wrh_realloc_df_not_satisfy_new_sku_not_satisfy = SkuWarehouseReallocation._sku_priority_allocation(
                    use_data_df=sku_wrh_realloc_df_not_satisfy_new_sku_not_satisfy,
                    stock_sum_col='warehouse_sku_inv',
                    priority_rank_col='store_cate1_priority',
                    alloc_col='alloc_cnt',
                    alloc_reason=",新品库存不足按照大类门店优先级分配"
                )[self.output_columns]
                sku_wrh_realloc_df_not_satisfy_new_sku_not_satisfy = sku_wrh_realloc_df_not_satisfy_new_sku_not_satisfy.reset_index(
                    drop=False).set_index(
                    ['store_code', 'goods_code'])
            else:
                sku_wrh_realloc_df_not_satisfy_new_sku_not_satisfy = _TMP_df.copy(deep=True)

            # 汇总新品分配的完整结果
            store_wrh_alloc_new_sku_alloc_df = pd.concat([
                sku_wrh_realloc_df_not_satisfy_new_sku_satisfy,
                sku_wrh_realloc_df_not_satisfy_new_sku_not_satisfy
            ], ignore_index=False)
            # 汇总每个sku，用于计算剩余可用库存
            store_wrh_new_sku_alloc_sum_df = store_wrh_alloc_new_sku_alloc_df.groupby(
                level='goods_code')['warehouse_alloc_cnt'].sum().to_frame()
        else:
            store_wrh_alloc_new_sku_alloc_df = _TMP_df.copy(deep=True)
            store_wrh_new_sku_alloc_sum_df = _TMP_df.copy(deep=True)
        store_wrh_new_sku_alloc_sum_df.rename(columns={'warehouse_alloc_cnt': 'new_sku_alloc_sum'}, inplace=True)

        # 2.2 对老品进行重分配
        if len(sku_wrh_realloc_df_not_satisfy_old_sku) == 0:
            return pd.concat([sku_wrh_realloc_df_all_satisfy, store_wrh_alloc_new_sku_alloc_df], ignore_index=False)
        else:
            # 先减去新品已经占用的库存
            if len(store_wrh_new_sku_alloc_sum_df) > 0:
                # 计算新品分配后的库存可用数量
                sku_wrh_realloc_df_not_satisfy_old_sku = pd.merge(
                    left=sku_wrh_realloc_df_not_satisfy_old_sku,
                    right=store_wrh_new_sku_alloc_sum_df,
                    left_index=True,
                    right_index=True,
                    how="left"
                )
                sku_wrh_realloc_df_not_satisfy_old_sku['new_sku_alloc_sum'].fillna(0.0, inplace=True)
            else:
                sku_wrh_realloc_df_not_satisfy_old_sku['new_sku_alloc_sum'] = 0.0
            # 对新品分配后的库存进行重分配
            sku_wrh_realloc_df_not_satisfy_old_sku["warehouse_sku_inv_res"] = \
                sku_wrh_realloc_df_not_satisfy_old_sku[
                    'warehouse_sku_inv'] - \
                sku_wrh_realloc_df_not_satisfy_old_sku[
                    'new_sku_alloc_sum']

            sku_wrh_realloc_df_not_satisfy_old_sku.fillna(
                {'store_sku_inv': 0.0, 'store_sku_avg_day_sale': 0.0},
                inplace=True
            )
            # 计算单品剩余可售天数
            sku_wrh_realloc_df_not_satisfy_old_sku['store_sku_available_sale_day'] = np.divide(
                sku_wrh_realloc_df_not_satisfy_old_sku['store_sku_inv'],
                sku_wrh_realloc_df_not_satisfy_old_sku['store_sku_avg_day_sale']
            )

            # 若单品综合日均销=0，则将可售天数取100
            sku_wrh_realloc_df_not_satisfy_old_sku['store_sku_available_sale_day'] = np.where(
                sku_wrh_realloc_df_not_satisfy_old_sku['store_sku_avg_day_sale'] <= 0.0,
                100,
                sku_wrh_realloc_df_not_satisfy_old_sku['store_sku_available_sale_day']
            )
            # 单品门店优先级: 0.4 * 单品剩余可售天数 + 0.6 * 单品综合日均销
            # TODO 此处权重后期需调整为通过config传入
            sku_wrh_realloc_df_not_satisfy_old_sku['store_sku_score'] = sku_wrh_realloc_df_not_satisfy_old_sku[
                                                                            'store_sku_available_sale_day'] * 0.4 + \
                                                                        sku_wrh_realloc_df_not_satisfy_old_sku[
                                                                            'store_sku_avg_day_sale'] * 0.6

            sku_wrh_realloc_df_not_satisfy_old_sku.reset_index('store_code', drop=False, inplace=True)
            # 数据排序
            sku_wrh_realloc_df_not_satisfy_old_sku.sort_values(
                by=['store_sku_score', 'store_code'],
                ascending=False, inplace=True
            )
            sku_wrh_realloc_df_not_satisfy_old_sku[
                'store_sku_priority'] = np.arange(len(sku_wrh_realloc_df_not_satisfy_old_sku)) + 1
            # 1. 先按照各店需求的50%分配:roundup(需求量*0.5/规格,0)*规格
            sku_wrh_realloc_df_not_satisfy_old_sku['alloc_cnt_first'] = np.multiply(
                np.ceil(
                    np.divide(
                        sku_wrh_realloc_df_not_satisfy_old_sku['alloc_cnt'] * 0.5,
                        sku_wrh_realloc_df_not_satisfy_old_sku['conversion_qty']
                    )
                ),
                sku_wrh_realloc_df_not_satisfy_old_sku['conversion_qty']
            )

            sku_wrh_realloc_df_not_satisfy_old_sku["alloc_cnt_first_sum"] = \
                sku_wrh_realloc_df_not_satisfy_old_sku.groupby("goods_code"
                                                               )["alloc_cnt_first"].transform("sum")

            # 判断第一步分配是否存在有不足分配的情况，如存在，则基于单品门店优先级进行排序分配，否则进行下一轮
            sku_wrh_realloc_df_not_satisfy_old_sku_satisfy = sku_wrh_realloc_df_not_satisfy_old_sku[
                sku_wrh_realloc_df_not_satisfy_old_sku[
                    'alloc_cnt_first_sum'] <= \
                sku_wrh_realloc_df_not_satisfy_old_sku[
                    'warehouse_sku_inv_res']
                ]

            sku_wrh_realloc_df_not_satisfy_old_sku_not_satisfy = sku_wrh_realloc_df_not_satisfy_old_sku[
                sku_wrh_realloc_df_not_satisfy_old_sku[
                    'alloc_cnt_first_sum'] > \
                sku_wrh_realloc_df_not_satisfy_old_sku[
                    'warehouse_sku_inv_res']
                ]
            # 若第一轮即不满足，则按照单品门店优先级进行分配
            if len(sku_wrh_realloc_df_not_satisfy_old_sku_not_satisfy) > 0:
                sku_wrh_realloc_df_not_satisfy_old_sku_not_satisfy = SkuWarehouseReallocation._sku_priority_allocation(
                    use_data_df=sku_wrh_realloc_df_not_satisfy_old_sku_not_satisfy,
                    stock_sum_col='warehouse_sku_inv_res',
                    alloc_col='alloc_cnt_first',
                    priority_rank_col='store_sku_priority',
                    alloc_reason=',非新品第1轮按单品优先级分配'
                )[self.output_columns]
            else:
                sku_wrh_realloc_df_not_satisfy_old_sku_not_satisfy = _TMP_df.copy(deep=True)

            # 假设第一轮满足，则进行第二论分配
            if len(sku_wrh_realloc_df_not_satisfy_old_sku_satisfy) > 0:
                sku_wrh_realloc_df_not_satisfy_old_sku_satisfy['warehouse_sku_inv_first_res'] = \
                    sku_wrh_realloc_df_not_satisfy_old_sku_satisfy['warehouse_sku_inv_res'] - \
                    sku_wrh_realloc_df_not_satisfy_old_sku_satisfy['alloc_cnt_first_sum']
                sku_wrh_realloc_df_not_satisfy_old_sku_satisfy["alloc_reason"] += \
                    ",非新品第1轮一半初始分配" + \
                    sku_wrh_realloc_df_not_satisfy_old_sku_satisfy['alloc_cnt_first'].astype('str')

                store_wrh_realloc_df_not_satisfy_old_sku_second_alloc = \
                    SkuWarehouseReallocation._sku_priority_allocation(
                        use_data_df=sku_wrh_realloc_df_not_satisfy_old_sku_satisfy,
                        stock_sum_col='warehouse_sku_inv_first_res',
                        alloc_col='alloc_cnt_first',
                        priority_rank_col='store_sku_priority',  # 此处第一轮和第二轮需求量一样
                        alloc_reason=',非新品第2轮按单品优先级分配'
                    )[self.output_columns]
                # 将两轮结果汇总得到最终结果
                store_wrh_realloc_df_not_satisfy_old_sku_second_alloc.rename(
                    columns={'warehouse_alloc_cnt': 'alloc_cnt_second'}, inplace=True
                )
                sku_wrh_realloc_df_not_satisfy_old_sku_satisfy = sku_wrh_realloc_df_not_satisfy_old_sku_satisfy.reset_index(
                    drop=False).set_index(['store_code', 'goods_code'])
                store_wrh_realloc_df_not_satisfy_old_sku_satisfy = pd.merge(
                    left=sku_wrh_realloc_df_not_satisfy_old_sku_satisfy,
                    right=store_wrh_realloc_df_not_satisfy_old_sku_second_alloc[
                        ['alloc_cnt_second', 'warehouse_alloc_reason']],
                    left_index=True,
                    right_index=True,
                    how='left',
                )
                store_wrh_realloc_df_not_satisfy_old_sku_satisfy['warehouse_alloc_cnt'] = \
                    store_wrh_realloc_df_not_satisfy_old_sku_satisfy['alloc_cnt_first'] + \
                    store_wrh_realloc_df_not_satisfy_old_sku_satisfy['alloc_cnt_second']
                store_wrh_realloc_df_not_satisfy_old_sku_satisfy = store_wrh_realloc_df_not_satisfy_old_sku_satisfy[
                    self.output_columns]
            else:
                store_wrh_realloc_df_not_satisfy_old_sku_satisfy = _TMP_df.copy(deep=True)
            # 合并结果
            store_wrh_old_sku_realloc_df = pd.concat(
                [sku_wrh_realloc_df_not_satisfy_old_sku_not_satisfy,
                 store_wrh_realloc_df_not_satisfy_old_sku_satisfy],
                ignore_index=False
            )
            sku_warehouse_reallocation_df = pd.concat(
                [sku_wrh_realloc_df_all_satisfy, store_wrh_alloc_new_sku_alloc_df, store_wrh_old_sku_realloc_df],
                ignore_index=False
            )

            if len(sku_warehouse_reallocation_df) != len(sku_init_alloc_df):
                logger.error(
                    f"SkuWarehouseReallocation calculate error:data count is {len(sku_warehouse_reallocation_df)}"
                    f",not equal to base table count {len(input_data_df)}"
                )
            return sku_warehouse_reallocation_df

    @staticmethod
    @logger.catch
    def _sku_priority_allocation(use_data_df: pd.DataFrame,
                                 stock_sum_col: str,
                                 alloc_col: str,
                                 priority_rank_col: str,
                                 alloc_reason: str) -> pd.DataFrame:
        """
        用于在已知可用库存不足以全部分配时，按照优先级顺序进行分配
        注意：此处在分配前后，都需要用商品规格对计算结果进行处理，保证最终出数符合中包标准
        """
        input_data_df = use_data_df.copy(deep=True)
        assert stock_sum_col in input_data_df.columns, f"{stock_sum_col} not in dataframe"
        assert alloc_col in input_data_df.columns, f"{alloc_col} not in dataframe"
        assert priority_rank_col in input_data_df.columns, f"{priority_rank_col} not in dataframe"
        assert 'conversion_qty' in input_data_df.columns, f"conversion_qty not in dataframe"
        # 先对分配结果进行标准中包计算
        input_data_df[alloc_col] = np.multiply(
            np.ceil(
                np.divide(input_data_df[alloc_col].values, input_data_df['conversion_qty'].values)
            ),
            input_data_df['conversion_qty'].values
        )
        # 按照优先级排序
        input_data_df.sort_values(by=[priority_rank_col], ascending=True)
        # 计算各个商品需求量的累加和
        input_data_df[alloc_col + '_cumsum'] = input_data_df.groupby(level='goods_code')[alloc_col].cumsum()
        # 对每个sku内部基于初始分配量进行分配
        input_data_df[f'{alloc_col}_res'] = input_data_df[stock_sum_col].values - input_data_df[
            f'{alloc_col}_cumsum'].values
        # 还原每步分配前一步剩余库存量
        input_data_df[f'{alloc_col}_res_last'] = input_data_df[f'{alloc_col}_res'] + input_data_df[alloc_col]
        # 将每步分配前已经库存小于0的数据全部置为0
        input_data_df[f'{alloc_col}_res_last'].clip(lower=0.0, inplace=True)
        # 计算最终的分配量
        input_data_df['alloc_cnt_warehouse'] = np.min(
            input_data_df[[alloc_col, f'{alloc_col}_res_last']],
            axis=1
        )
        # 再次对分配结果进行规格处理，需要注意的是此处采用下舍取整，保证结果一定不会超过可用库存总量
        input_data_df['alloc_cnt_warehouse'] = np.multiply(
            np.floor(
                np.divide(input_data_df['alloc_cnt_warehouse'].values, input_data_df['conversion_qty'].values)
            ),
            input_data_df['conversion_qty']
        )
        input_data_df = input_data_df.reset_index(drop=False).set_index(['store_code', 'goods_code'])
        input_data_df['alloc_reason'] += alloc_reason + np.where(
            input_data_df['alloc_cnt_warehouse'] > 0,
            np.round(input_data_df[stock_sum_col] + 1e-8).astype('int').astype('str') + "," +
            np.round(input_data_df["alloc_cnt_warehouse"] + 1e-8).astype('int').astype('str'),
            ",库存不足分配为0"
        )

        input_data_df.rename(
            columns={"alloc_cnt_warehouse": "warehouse_alloc_cnt", "alloc_reason": "warehouse_alloc_reason"},
            inplace=True
        )

        return input_data_df


sku_warehouse_reallocation = SkuWarehouseReallocation()
