# -*- encoding: utf-8 -*-
"""
@Time        : 2022/2/23 3:41 PM
@Author      : miniso
@Description : 初始分配计算模块
"""
import numpy as np
import pandas as pd
from cate4_replenishment_space import cate4_replenishment_space
from common import CommonObject
from data_load import mission_info_data_load
from data_load import sku_display_minimum_data_load
from data_load import stock_data_load
from sku_init_replenishment import sku_init_replenishment
from loguru import logger


class SkuInitAllocation(CommonObject):

    def __init__(self):
        super().__init__()
        # 初始分配结果
        self.sku_init_allocation_df = None
        self.output_columns = ['alloc_cnt', 'alloc_reason', 'new_arrival', 'conversion_qty', 'store_sku_inv',
                               'store_sku_avg_day_sale', 'sales_pred_d_avg_store_sku_fixed',
                               'sales_pred_d_avg_store_sku_fixed_fill_with_avg', 'sku_minimum',
                               'sales_pred_d_avg_store_cate4_fixed',
                               'warehouse_sku_inv', 'cate_level4_code', 'sku_norm_cnt', 'cate_level1_code',
                               'new_store', 'store_cate1_avg_day_sale',
                               'cate4_need', 'cate4_no_need', 'cate4_replenishment_space', 'mainwarehouse',
                               'series_sku_tag',
                               'sku_init_replenishment', 'sku_init_replenishment_upper',
                               'sku_qty', 'pcs_qty', 'store_sku_display_minimum', 'store_sku_priority',
                               'store_sku_week_fix_avg_day_sale',
                               'top_sales', 'whether_abolish', 'wrh_sku_display_minimum',
                               'store_sku_inv_base', 'warehouse_sku_inv_base', 'sku_init_replenishment_base',
                               'sku_init_replenishment_upper_base',
                               'cate4_replenishment_space_base', 'sku_minimum_cate4_sum', 'sale_limit',
                               'sale_day_limit']

    @logger.catch
    def calculate(self) -> pd.DataFrame:
        if self.sku_init_allocation_df is None:
            # 数据准备
            sku_init_allocation_df = self.data_prepare()
            # 初始分配计算
            self.sku_init_allocation_df = self.sku_init_allocation_cal(sku_init_allocation_df)
            self.sku_init_allocation_df = self.sku_init_allocation_df[self.output_columns]
            self.check_amount(self.sku_init_allocation_df, len(sku_init_allocation_df))
            self.check_zero(self.sku_init_allocation_df, 'alloc_cnt')
            self.check_negative(self.sku_init_allocation_df, 'alloc_cnt')
        logger.info("SkuInitAllocation finished!")
        return self.sku_init_allocation_df

    @logger.catch
    def data_prepare(self) -> pd.DataFrame:
        # 首先计算前置依赖
        # 细类补货空间计算
        cate4_replenishment_space_df = cate4_replenishment_space.calculate()
        # sku初始补货量,index=[store_code,cate_level4_code,goods_code]
        sku_init_replenishment_df = sku_init_replenishment.calculate()
        logger.info("start run SkuInitAllocation")
        # 加载相关依赖
        # 商品标签，用于优先级计算
        store_sku_label = mission_info_data_load.label_info()[
            ['top_sales', 'new_arrival', 'replenishment_lifecycle']]
        store_sku_display_minimum_df = sku_display_minimum_data_load.sku_display_minimum()[['final_num']]  # 门店陈列保底量
        wrh_sku_display_minimum_df = sku_display_minimum_data_load.sku_display_minimum_wrh()[
            ['final_num']]  # 门店陈列保底量
        store_sku_minimum_df = mission_info_data_load.minimum_info()[['sku_minimum', 'series_minimum']]
        warehouse_sku_stock_df = stock_data_load.warehouse_sku_stock()[['inv_available']]  # 仓库商品库存读取
        area_dim_df = mission_info_data_load.area_dim()[['new_store', 'mainwarehouse', 'store_level']]
        store_cate1_sale_df = mission_info_data_load.store_cate1_sale()[['sales_avg']]  # 门店大类日均销

        logger.info("SkuInitAllocation data load finished!")

        # 一、数据拼接
        sku_init_allocation_df = pd.merge(left=sku_init_replenishment_df,
                                          right=cate4_replenishment_space_df,
                                          left_index=True,
                                          right_index=True,
                                          how='inner'
                                          )
        # 保底数据
        sku_init_allocation_df.reset_index('cate_level4_code', drop=False, inplace=True)
        sku_init_allocation_df[['sku_minimum', 'series_sku_tag']] = store_sku_minimum_df[
            ['sku_minimum', 'series_minimum']]
        # 商品标签
        sku_init_allocation_df[['top_sales', 'new_arrival', 'replenishment_lifecycle']] = store_sku_label[
            ['top_sales', 'new_arrival', 'replenishment_lifecycle']]

        # 废除期商品
        sku_init_allocation_df['whether_abolish'] = 0
        sku_init_allocation_df.loc[sku_init_allocation_df['replenishment_lifecycle'] == '60', 'whether_abolish'] = 1
        del sku_init_allocation_df['replenishment_lifecycle']
        # 陈列保底
        sku_init_allocation_df['store_sku_display_minimum'] = store_sku_display_minimum_df['final_num']
        # 匹配仓库库存数据，分配要对仓库库存数据进行过滤
        sku_init_allocation_df.reset_index('goods_code', drop=False, inplace=True)
        sku_init_allocation_df[['new_store', 'mainwarehouse', 'store_level']] = area_dim_df[
            ['new_store', 'mainwarehouse', 'store_level']]
        # 基于不同门店等级，设置不同销量上限及对应可售天数上限，用于畅销保底控制
        sale_limit = {'S': 1.5, 'A+': 1.5, 'A': 1.5, 'B': 1.5, 'C': 1.5, 'D': 1.1, 'E': 1.1, 'F': 0.8}
        sale_day_limit = {'S': 7, 'A+': 7, 'A': 7, 'B': 7, 'C': 7, 'D': 10, 'E': 10, 'F': 15}
        sku_init_allocation_df['sale_limit'] = sku_init_allocation_df['store_level'].map(sale_limit, na_action='ignore')
        sku_init_allocation_df['sale_day_limit'] = sku_init_allocation_df['store_level'].map(sale_day_limit,
                                                                                             na_action='ignore')
        # 匹配门店大类日均销
        sku_init_allocation_df.set_index('cate_level1_code', append=True, inplace=True)
        sku_init_allocation_df['store_cate1_avg_day_sale'] = store_cate1_sale_df['sales_avg']
        sku_init_allocation_df = sku_init_allocation_df.reset_index().set_index(['mainwarehouse', 'goods_code'])
        # 仓库库存
        sku_init_allocation_df['warehouse_sku_inv'] = warehouse_sku_stock_df['inv_available']
        # 仓库陈列保底
        sku_init_allocation_df['wrh_sku_display_minimum'] = wrh_sku_display_minimum_df['final_num']

        # 修改数据类型
        sku_init_allocation_df = sku_init_allocation_df.astype(
            {'sku_minimum': float, 'top_sales': int, 'new_arrival': int},
            errors="ignore"
        )
        # 填充缺失值
        sku_init_allocation_df.fillna(
            {
                'cate4_replenishment_space': 0.0, 'sku_minimum': 0.0,
                'top_sales': 0, 'new_arrival': 0, 'replenishment_lifecycle': 0,
                'warehouse_sku_inv': 0.0, 'store_sku_display_minimum': 0.0, 'wrh_sku_display_minimum': 0.0,
                'new_store': 0, 'store_cate1_avg_day_sale': 0.0

            },
            inplace=True
        )
        sku_init_allocation_df = sku_init_allocation_df.reset_index(drop=False).set_index(
            ['store_code', 'goods_code', 'cate_level4_code'])

        # 计算单品优先级,用于分配计算
        # 畅销品>新品>正常商品，当优先级一致时，按照预测销量、仓库日均销、商品序号三个字段依次排序
        sku_init_allocation_df['goods_code_bysort'] = sku_init_allocation_df.index.get_level_values('goods_code')
        sku_init_allocation_df.sort_values(
            by=['top_sales', 'new_arrival',
                'sales_pred_d_avg_store_sku_fixed', 'wrh_sku_avg_day_sale',
                'goods_code_bysort'],
            ascending=False, inplace=True
        )
        sku_init_allocation_df['store_sku_priority'] = sku_init_allocation_df.groupby(
            level=['store_code', 'cate_level4_code']).cumcount() + 1
        del sku_init_allocation_df['goods_code_bysort']
        # 增加几个冗余字段，便于下文计算
        sku_init_allocation_df[
            ['store_sku_inv_base', 'warehouse_sku_inv_base', 'sku_init_replenishment_base',
             'sku_init_replenishment_upper_base',
             'cate4_replenishment_space_base']] = \
            sku_init_allocation_df[[
                'store_sku_inv', 'warehouse_sku_inv', 'sku_init_replenishment', 'sku_init_replenishment_upper',
                'cate4_replenishment_space']]
        return sku_init_allocation_df

    @logger.catch
    def sku_init_allocation_cal(self, input_data_df: pd.DataFrame) -> pd.DataFrame:
        """
           初始分配完整计算环节
        """
        sku_init_allocation_df = input_data_df.copy(deep=True)
        # 0. 保底计算
        # 先进行保底商品分配,并计算分配后的剩余空间
        # 注意：触发保底的条件:
        # 1. 保底量>=0, 2.细类定标>=0,3.仓库库存>0,4.保底量>门店库存
        sku_init_allocation_df['sku_alloc_minimum'] = np.where(
            (sku_init_allocation_df['sku_minimum'].values > 0.0) &
            (sku_init_allocation_df['sku_qty'].values > 0.0) &
            (sku_init_allocation_df['pcs_qty'].values > 0.0) &
            (sku_init_allocation_df['warehouse_sku_inv'].values > 0.0) &
            (sku_init_allocation_df['sku_minimum'].values > sku_init_allocation_df['store_sku_inv'].values),
            sku_init_allocation_df['sku_minimum'] - sku_init_allocation_df['store_sku_inv'],
            0.0
        )
        # 按规格向上取整
        sku_init_allocation_df['sku_alloc_minimum'] = np.ceil(
            sku_init_allocation_df['sku_alloc_minimum'].values / sku_init_allocation_df['conversion_qty'].values
        ) * sku_init_allocation_df['conversion_qty'].values

        sku_init_allocation_df['sku_alloc_minimum_sum'] = sku_init_allocation_df.groupby(
            level=['store_code', 'cate_level4_code'])['sku_alloc_minimum'].transform('sum')
        sku_init_allocation_df['cate4_rep_space_no_min'] = sku_init_allocation_df[
                                                               'cate4_replenishment_space'].values - \
                                                           sku_init_allocation_df[
                                                               'sku_alloc_minimum_sum'].values

        # 针对以下情况将剩余空间直接赋值为0
        # 1.因为保底新增1个补货空间的情况，保底计算后直接赋值为0
        # 2.保底分配后，补货空间小于0的情况
        sku_init_allocation_df['cate4_rep_space_no_min'] = np.where(
            (sku_init_allocation_df['cate4_rep_space_no_min'] < 0.0) |
            (
                    (sku_init_allocation_df['cate4_replenishment_space_base'] == 1) &
                    (sku_init_allocation_df['sku_minimum_cate4_sum'] > 0.0)
            ),
            0.0,
            sku_init_allocation_df['cate4_rep_space_no_min']
        )
        # 计算保底商品剩余分配量
        sku_init_allocation_df['sku_init_rep_no_min'] = sku_init_allocation_df['sku_init_replenishment'].values - \
                                                        sku_init_allocation_df['sku_alloc_minimum'].values
        sku_init_allocation_df['sku_init_rep_no_min'].clip(lower=0.0, inplace=True)
        sku_init_allocation_df['sku_init_rep_no_min_upper'] = sku_init_allocation_df[
                                                                  'sku_init_replenishment_upper'].values - \
                                                              sku_init_allocation_df['sku_alloc_minimum'].values
        sku_init_allocation_df['sku_init_rep_no_min_upper'].clip(lower=0.0, inplace=True)
        # 当门店陈列有数据时，采用门店陈列，否则使用仓库陈列
        sku_init_allocation_df['display_minimum'] = np.where(
            sku_init_allocation_df['store_sku_display_minimum'] > 0.0,
            sku_init_allocation_df['store_sku_display_minimum'],
            sku_init_allocation_df['wrh_sku_display_minimum']
        )

        # 1. 基础量计算
        # 首先筛选细类空间大于0的数据
        sku_init_allocation_cate4space_df = sku_init_allocation_df[
            sku_init_allocation_df['cate4_rep_space_no_min'] > 0.0]
        # 首先过滤掉废除期和禁配的商品，禁配已经前置删除，仅考虑废除期商品
        # 之后按照仓库库存数据进行过滤，仅对有库存数据进行分配
        sku_init_allocation_cate4space_df_noabolish = sku_init_allocation_cate4space_df[
            (sku_init_allocation_cate4space_df['whether_abolish'] != 1) &
            (sku_init_allocation_cate4space_df['warehouse_sku_inv'] > 0)
            ]
        # 按照优先级排序
        # 计算各个商品需求量的累加和
        sku_init_allocation_df_base = SkuInitAllocation._sku_priority_init_allocation(
            use_data_df=sku_init_allocation_cate4space_df_noabolish,
            alloc_col='sku_init_rep_no_min',
            space_col='cate4_rep_space_no_min',
            priority_rank_col='store_sku_priority',
            last_half_fill=True
        )
        sku_init_allocation_cate4space_df['init_alloc_base'] = sku_init_allocation_df_base['sku_init_rep_no_min_cal']
        sku_init_allocation_cate4space_df['init_alloc_base'].fillna(0.0, inplace=True)

        # 2.细类空间补足
        # 计算剩余细类空间
        sku_init_allocation_cate4space_df['init_alloc_base_sum'] = sku_init_allocation_cate4space_df.groupby(
            level=['store_code', 'cate_level4_code'])['init_alloc_base'].transform('sum')
        sku_init_allocation_cate4space_df['cate4_replenishment_space_res'] = sku_init_allocation_cate4space_df[
                                                                                 'cate4_rep_space_no_min'].values - \
                                                                             sku_init_allocation_cate4space_df[
                                                                                 'init_alloc_base_sum'].values
        # 过滤掉新品和已经达到补货空间的数据
        sku_init_allocation_cate4space_df_havespace = sku_init_allocation_cate4space_df[
            (sku_init_allocation_cate4space_df['cate4_replenishment_space_res'] > 0.0) &
            (sku_init_allocation_cate4space_df['new_arrival'] != 1)
            ]

        # 计算当前每个商品还可以分配的量
        sku_init_allocation_cate4space_df_havespace['sku_alloc_res'] = \
            sku_init_allocation_cate4space_df_havespace['sku_init_rep_no_min_upper'].values - \
            sku_init_allocation_cate4space_df_havespace['init_alloc_base'].values

        # 按照同样的优先级逻辑进行分配
        sku_init_allocation_cate4spacefill = SkuInitAllocation._sku_priority_init_allocation(
            use_data_df=sku_init_allocation_cate4space_df_havespace,
            alloc_col='sku_alloc_res',
            space_col='cate4_replenishment_space_res',
            priority_rank_col='store_sku_priority'
        )
        sku_init_allocation_cate4space_df['sku_init_space4fill'] = sku_init_allocation_cate4spacefill[
            'sku_alloc_res_cal']
        sku_init_allocation_df[['init_alloc_base', 'sku_init_space4fill']] = sku_init_allocation_cate4space_df[
            ['init_alloc_base', 'sku_init_space4fill']]

        sku_init_allocation_df.fillna({'init_alloc_base': 0.0, 'sku_init_space4fill': 0.0}, inplace=True)
        # 基础量
        sku_init_allocation_df['init_alloc_base_minimum'] = sku_init_allocation_df['sku_alloc_minimum'].values + \
                                                            sku_init_allocation_df['init_alloc_base'].values
        # 空间补足量
        sku_init_allocation_df['sku_init_alloc_cnt'] = sku_init_allocation_df['init_alloc_base_minimum'].values + \
                                                       sku_init_allocation_df['sku_init_space4fill'].values

        # 3. top300 保底陈列检查
        # 计算实际保底陈列量
        sku_init_allocation_df['store_sku_pred_sale_sum_tmp'] = np.select(
            condlist=[
                sku_init_allocation_df['sales_pred_d_avg_store_sku_fixed_fill_with_avg'] == 0,
                sku_init_allocation_df['sales_pred_d_avg_store_sku_fixed_fill_with_avg'].between(0, 2,
                                                                                                 inclusive='right'),
                sku_init_allocation_df['sales_pred_d_avg_store_sku_fixed_fill_with_avg'] > 2
            ],
            choicelist=[
                0,
                28,
                21
            ]
        ) * sku_init_allocation_df['sales_pred_d_avg_store_sku_fixed_fill_with_avg']
        sku_init_allocation_df['store_sku_pred_sale_sum_tmp'].fillna(0.0, inplace=True)
        # 实际保底陈列量
        sku_init_allocation_df['store_sku_display_actual_minimum'] = np.where(
            (sku_init_allocation_df['sales_pred_d_avg_store_sku_fixed_fill_with_avg'] == 0) | (
                sku_init_allocation_df['sales_pred_d_avg_store_sku_fixed_fill_with_avg'].isnull()),
            sku_init_allocation_df['display_minimum'] * 0.5,
            np.min(
                sku_init_allocation_df[['store_sku_pred_sale_sum_tmp', 'display_minimum']],
                axis=1
            )
        )
        del sku_init_allocation_df['store_sku_pred_sale_sum_tmp']
        # 除不需要细类外，其他均将初始分配量+当前门店库存提升到实际陈列量
        sku_init_allocation_df['sku_init_alloc_cnt_display'] = np.where(
            (sku_init_allocation_df['cate4_no_need'] != 1) &
            (
                    sku_init_allocation_df['sku_init_alloc_cnt'].values +
                    sku_init_allocation_df['store_sku_inv'].values <
                    sku_init_allocation_df['store_sku_display_actual_minimum'].values
            ),
            np.ceil(
                (sku_init_allocation_df['store_sku_display_actual_minimum'].values - sku_init_allocation_df[
                    'store_sku_inv'].values) /
                sku_init_allocation_df['conversion_qty'].values
            ) * sku_init_allocation_df['conversion_qty'].values,
            sku_init_allocation_df['sku_init_alloc_cnt'].values
        )
        sku_init_allocation_df['sku_init_alloc_cnt_display'].fillna(0.0, inplace=True)
        # 4. 系列性铺货检查
        # 当前门店库存量不足一个中包，就分配一个，其他的不做操作
        sku_init_allocation_df['sku_init_alloc_cnt_display_series'] = np.where(
            (sku_init_allocation_df['series_sku_tag'] == 1) &
            (sku_init_allocation_df['sku_init_alloc_cnt_display'].values > 0.0) &
            (sku_init_allocation_df['sku_init_alloc_cnt_display'].values +
             sku_init_allocation_df['store_sku_inv'].values <
             sku_init_allocation_df['conversion_qty'].values * 0.5
             ),
            sku_init_allocation_df['conversion_qty'].values,
            sku_init_allocation_df['sku_init_alloc_cnt_display'],
        )
        sku_init_allocation_df['sku_init_alloc_cnt_display_series'].fillna(0.0, inplace=True)

        # 5. 畅销品保底检查
        # 计算可售天数
        sku_init_allocation_df['store_sku_sale_day'] = (sku_init_allocation_df['store_sku_inv'].values + \
                                                        sku_init_allocation_df[
                                                            'sku_init_alloc_cnt_display_series'].values) / \
                                                       sku_init_allocation_df[
                                                           'sales_pred_d_avg_store_sku_fixed_fill_with_avg'].values
        # 进入畅销保底的条件
        # a.预测>=销量限制
        # b.日均销>0
        # c.可售天数>0 and < 可售天数限制
        # 计算公式= 规格取整(预测*可售天数限制-门店库存)
        sku_init_allocation_df['sku_init_alloc_cnt_display_series_top'] = np.where(
            (sku_init_allocation_df['sales_pred_d_avg_store_sku_fixed_fill_with_avg'] >= sku_init_allocation_df[
                'sale_limit']) &
            (sku_init_allocation_df['store_sku_week_fix_avg_day_sale'].values > 0.0) &
            (sku_init_allocation_df['store_sku_sale_day'].values > 0.0) &
            (sku_init_allocation_df['store_sku_sale_day'].values < sku_init_allocation_df['sale_day_limit'].values),
            np.ceil(
                (sku_init_allocation_df['sales_pred_d_avg_store_sku_fixed_fill_with_avg'].values *
                 sku_init_allocation_df['sale_day_limit'].values
                 -
                 sku_init_allocation_df['store_sku_inv'].values) / sku_init_allocation_df['conversion_qty'].values
            ) * sku_init_allocation_df['conversion_qty'].values,
            sku_init_allocation_df['sku_init_alloc_cnt_display_series'].values
        )
        sku_init_allocation_df['sku_init_alloc_cnt_display_series_top'].fillna(0.0, inplace=True)
        sku_init_allocation_df.rename(columns={'sku_init_alloc_cnt_display_series_top': 'alloc_cnt'}, inplace=True)
        if len(sku_init_allocation_df) != len(input_data_df):
            logger.error(
                f"SkuInitAllocation calculate error:data count is {len(sku_init_allocation_df)}"
                f",not equal to base table count {len(input_data_df)}"
            )
        sku_init_allocation_df['alloc_reason'] = \
            '补货空间:' + np.round(sku_init_allocation_df['cate4_replenishment_space'] + 1e-8).astype('int').astype('str') + \
            ",需要:" + np.where(sku_init_allocation_df['cate4_need'] == 1,
                              '✓',
                              np.where(sku_init_allocation_df['cate4_no_need'] == 1, '×', '空')
                              ) + \
            '_初始补货:基础' + np.round(sku_init_allocation_df['sku_init_replenishment'] + 1e-8).astype('int').astype('str') + \
            ',上限' + np.round(sku_init_allocation_df['sku_init_replenishment_upper'] + 1e-8).astype('int').astype(
                'str') + \
            '_初始分配:门店单品优先级' + sku_init_allocation_df['store_sku_priority'].fillna(0).astype('int').astype(
                'str') + \
            ',畅销品' + np.where(sku_init_allocation_df['top_sales'] == 1, '✓', '×') + \
            ',新品' + np.where(sku_init_allocation_df['new_arrival'] == 1, '✓', '×') + \
            ',销量' + np.round(sku_init_allocation_df['store_sku_avg_day_sale'] + 1e-8, 2).astype('str') + \
            ',保底' + np.round(sku_init_allocation_df['sku_alloc_minimum'] + 1e-8).astype('int').astype('str') + \
            ',基础' + np.round(sku_init_allocation_df['init_alloc_base_minimum'] + 1e-8).astype('int').astype('str') + \
            ',空间补足' + np.round(sku_init_allocation_df['sku_init_alloc_cnt'] + 1e-8).astype('int').astype('str') + \
            ',保底陈列' + np.round(sku_init_allocation_df['sku_init_alloc_cnt_display'] + 1e-8).astype('int').astype(
                'str') + \
            ',系列铺货' + np.round(sku_init_allocation_df['sku_init_alloc_cnt_display_series'] + 1e-8).astype('int').astype(
                'str') + \
            ',畅销保底' + np.round(sku_init_allocation_df['alloc_cnt'] + 1e-8).astype('int').astype('str')
        # 删除细类索引,仅保留store_code*goods_code
        sku_init_allocation_df.reset_index('cate_level4_code', drop=False, inplace=True)
        return sku_init_allocation_df

    @staticmethod
    @logger.catch
    def _sku_priority_init_allocation(
            use_data_df: pd.DataFrame,
            alloc_col: str,
            space_col: str,
            priority_rank_col: str,
            last_half_fill: bool = False
    ) -> pd.DataFrame:
        """
            基于优先级对细类补货空间进行填充
        :param use_data_df:要处理的数据
        :param alloc_col:要分配的数量
        :param space_col:总共的空间
        :param priority_rank_col:商品之间排序的依据
        :param last_half_fill:最终一个商品若小于半包，则按照一个整包分配
        """
        # TODO:与重分配的优先级分配逻辑有些许不同，后期可考虑融合
        input_data_df = use_data_df.copy(deep=True)
        assert alloc_col in input_data_df.columns, f"{alloc_col} not in dataframe"
        assert space_col in input_data_df.columns, f"{space_col} not in dataframe"
        assert priority_rank_col in input_data_df.columns, f"{priority_rank_col} not in dataframe"
        assert 'conversion_qty' in input_data_df.columns, f"conversion_qty not in dataframe"
        # 排序
        input_data_df.sort_values(by=[priority_rank_col], ascending=True, inplace=True)
        # 分配
        input_data_df[f'{alloc_col}_cumsum'] = input_data_df.groupby(level=['store_code', 'cate_level4_code'])[
            alloc_col].cumsum()
        # 计算补货空间剩余量
        input_data_df[f'{alloc_col}_cumsum_res'] = input_data_df[space_col] - input_data_df[f'{alloc_col}_cumsum']
        # 还原每步分配前一步剩余空间
        input_data_df[f'{alloc_col}_cumsum_res_last'] = input_data_df[f'{alloc_col}_cumsum_res'] + \
                                                        input_data_df[f'{alloc_col}']

        # 将每步分配前已经库存小于0的数据全部置为0
        input_data_df[f'{alloc_col}_cumsum_res_last'].clip(lower=0.0, inplace=True)
        # 计算最终的分配量
        input_data_df[f'{alloc_col}_cal'] = np.min(
            input_data_df[
                [f'{alloc_col}', f'{alloc_col}_cumsum_res_last']],
            axis=1
        )
        # 若最终分配的结果不到半个规格，那么就补到一个规格
        if last_half_fill:
            input_data_df[f'{alloc_col}_cal'] = np.where(
                input_data_df[f'{alloc_col}_cal'].between(0, input_data_df['conversion_qty'].values * 0.5,
                                                          inclusive=False),  # 开区间
                input_data_df['conversion_qty'],
                input_data_df[f'{alloc_col}_cal']
            )
        # 将结果按照中包四舍五入
        input_data_df[f'{alloc_col}_cal'] = np.round(
            input_data_df[f'{alloc_col}_cal'].values /
            input_data_df['conversion_qty'] + 1e-8) * input_data_df['conversion_qty']
        input_data_df = input_data_df[[f'{alloc_col}_cal']]
        return input_data_df


sku_init_allocation = SkuInitAllocation()
