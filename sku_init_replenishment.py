# -*- encoding: utf-8 -*-
"""
@Time        : 2022/2/23 3:37 PM
@Author      : miniso
@Description : 初始补货计算模块
"""
import numpy as np
import pandas as pd
from common import CommonObject
from data_load import cate4_demand_data_load
from data_load import cate4_max_days_data_load
from data_load import custom_conversion_qty_data_load
from data_load import goods_level_data_load
from data_load import mission_info_data_load
from data_load import stock_data_load
from data_load import upper_data_load
from loguru import logger


class SkuInitReplenishment(CommonObject):

    def __init__(self):
        super().__init__()
        self.sku_init_replenishment_df = None
        self.output_columns = ['sku_norm_cnt', 'sku_init_replenishment', 'sku_init_replenishment_upper',
                               'conversion_qty',
                               'store_sku_avg_day_sale', 'store_sku_week_fix_avg_day_sale', 'cate4_need',
                               'cate4_no_need',
                               'store_sku_inv',
                               'sales_pred_d_avg_store_sku_fixed', 'sales_pred_d_avg_store_sku_fixed_fill_with_avg',
                               'sku_qty', 'cate_level1_code', 'pcs_qty', 'new_arrival', 'wrh_sku_avg_day_sale']

    @logger.catch
    def calculate(self) -> pd.DataFrame:
        if self.sku_init_replenishment_df is None:
            # 1. 单品标准数量计算
            # 1.1 基础量计算
            logger.info("start run SkuInitReplenishment")
            base_table_df = mission_info_data_load.store_sku_dim()  # 任务涉及门店*sku维表
            cate4_display_info_df = mission_info_data_load.cate4_display_info()[['pcs_qty', 'sku_qty']]
            store_sku_sale_df = mission_info_data_load.store_sku_sale()[
                ['sales_avg_mixed_store_sku', 'sales_avg_store_sku', 'sales_avg_wrh_sku']]
            # store_sku_sale_df = sale_data_load.store_sku_sale()[['sales_avg_mixed_store_sku', 'sales_avg_store_sku']]
            # store_sku_avg_day_sale用于计算需求量，store_sku_week_fix_avg_day_sale用于需求上限条件判断
            dim_info = mission_info_data_load.goods_dim()[
                ['cate_level1_code', 'cate_level4_code', 'display_code', 'combine_price', 'conversion_qty']
            ]  # conversion_qty:中包规格
            dim_info = dim_info.astype({'conversion_qty': np.float16, 'display_code': str})
            custom_conversion_qty_df = custom_conversion_qty_data_load.custom_conversion_qty()[
                ['custom_conversion_qty']]
            # 使用用户自定义的中包规格替代系统输入
            dim_info['custom_conversion_qty'] = custom_conversion_qty_df['custom_conversion_qty']
            dim_info.loc[dim_info['custom_conversion_qty'].notnull(), 'conversion_qty'] = dim_info[
                'custom_conversion_qty']
            del dim_info['custom_conversion_qty']
            cate4_need_df = cate4_demand_data_load.cate4_need()
            cate4_need_df['cate4_need'] = 1
            cate4_need_df = cate4_need_df[['cate4_need']]
            cate4_no_need_df = cate4_demand_data_load.cate4_no_need()
            cate4_no_need_df['cate4_no_need'] = 1
            cate4_no_need_df = cate4_no_need_df[['cate4_no_need']]
            store_cate4_turnover_day = cate4_max_days_data_load.cate4_max_turnover_days()[
                ['turnover_days_avg']]  # 细类合理周转天数
            store_sku_stock_df = stock_data_load.store_sku_stock()[['inv']]  # 库存
            store_sku_sale_pred_df = mission_info_data_load.store_sku_sale_pred()[
                ['sales_pred_d_avg_store_sku_fixed']]  # 销量预测
            back_up_day_df = mission_info_data_load.back_up_days_info()[
                ['back_up_days', 'intransit_day', 'shipments_send_plan_interval']]  # 在途天数+发货间隔天数
            # 新品标签
            new_sku_label_df = mission_info_data_load.label_info()[['new_arrival']]

            # 单品标准数量上限
            sku_norm_cnt_upper_df = upper_data_load.sku_norm_cnt_upper()
            # 单品周转天数上限取数
            turnover_day_upper_df = upper_data_load.turnover_day_upper()
            upper_df = pd.merge(left=sku_norm_cnt_upper_df, right=turnover_day_upper_df,
                                on=['cate_level4_code', 'store_level'],
                                how='outer'
                                )
            # 门店等级标签
            store_level_df = mission_info_data_load.area_dim()[['store_level']]
            # 门店单品天数上限
            store_level_df['store_sku_day_upper_base'] = store_level_df['store_level'].map({
                'S': 28, 'A': 28, 'B': 35, 'C': 35, 'D': 35, 'E': 42, 'F': 42
            })
            logger.info("SkuInitReplenishment data load finished!")
            sku_norm_cnt_df = base_table_df.copy(deep=True)
            # sales_avg_mixed_store_sku:综合日均销，sales_avg_store_sku:星期修正的日均销,wrh_sku_avg_day_sale:仓库日均销
            sku_norm_cnt_df[['store_sku_avg_day_sale', 'store_sku_week_fix_avg_day_sale', 'wrh_sku_avg_day_sale']] = \
                store_sku_sale_df[['sales_avg_mixed_store_sku', 'sales_avg_store_sku', 'sales_avg_wrh_sku']]
            sku_norm_cnt_df['store_sku_avg_day_sale'].fillna(0.0, inplace=True)
            sku_norm_cnt_df = pd.merge(left=sku_norm_cnt_df,
                                       right=dim_info,
                                       left_index=True,
                                       right_index=True,
                                       how='left'
                                       )
            sku_norm_cnt_df = sku_norm_cnt_df.reset_index().set_index(['store_code', 'cate_level4_code'])
            sku_norm_cnt_df[['pcs_qty', 'sku_qty']] = cate4_display_info_df[['pcs_qty', 'sku_qty']]
            sku_norm_cnt_df['cate4_need'] = cate4_need_df['cate4_need']
            sku_norm_cnt_df['cate4_no_need'] = cate4_no_need_df['cate4_no_need']
            sku_norm_cnt_df = pd.merge(left=sku_norm_cnt_df,
                                       right=back_up_day_df,
                                       left_index=True,
                                       right_index=True,
                                       how='left'
                                       )
            sku_norm_cnt_df['back_up_days'] = sku_norm_cnt_df['back_up_days'].dt.days
            sku_norm_cnt_df.rename(columns={'shipments_send_plan_interval': 'delivery_interval_day'},
                                   inplace=True)
            sku_norm_cnt_df.fillna(0.0, inplace=True)  # 填充缺失值
            # 若单品对应细类日均销等于0，则第二项全为0
            sku_norm_cnt_df['store_sku_avg_day_sale_cate4_sum'] = sku_norm_cnt_df.groupby(
                level=['store_code', 'cate_level4_code'])['store_sku_avg_day_sale'].transform('sum')
            sku_norm_cnt_df['sku_norm_cnt'] = np.where(
                (sku_norm_cnt_df['sku_qty'].values == 0.0) |
                (sku_norm_cnt_df['pcs_qty'].values == 0.0),
                0.0,
                sku_norm_cnt_df['pcs_qty'].values / sku_norm_cnt_df['sku_qty'].values * 0.5
            ) + sku_norm_cnt_df['pcs_qty'].values * np.where(
                sku_norm_cnt_df['store_sku_avg_day_sale_cate4_sum'] <= 0.0, 0.0,
                sku_norm_cnt_df['store_sku_avg_day_sale'].values / sku_norm_cnt_df[
                    'store_sku_avg_day_sale_cate4_sum'].values * 0.5)
            del sku_norm_cnt_df['store_sku_avg_day_sale_cate4_sum']
            # 上限修正
            sku_norm_cnt_df.reset_index('cate_level4_code', drop=False, inplace=True)
            sku_norm_cnt_df = pd.merge(left=sku_norm_cnt_df,
                                       right=store_level_df,
                                       left_index=True,
                                       right_index=True,
                                       how='left')
            sku_norm_cnt_df['store_sku_day_upper_base'].fillna(35, inplace=True)
            sku_norm_cnt_df.reset_index(drop=False, inplace=True)
            sku_norm_cnt_df = pd.merge(left=sku_norm_cnt_df,
                                       right=upper_df,
                                       on=['cate_level4_code', 'store_level'],
                                       how='left'
                                       )
            sku_norm_cnt_df['sku_norm_cnt_upper'].fillna(0.0, inplace=True)
            # 用各个门店level的天数上限填充配置表缺失
            sku_norm_cnt_df['turnover_day_upper'].fillna(sku_norm_cnt_df['store_sku_day_upper_base'], inplace=True)
            # min(基础，max(单品标准数量上限,本店单品综合日均销*单品天数上限))
            sku_norm_cnt_df['sku_norm_cnt_upper_tmp'] = sku_norm_cnt_df['turnover_day_upper'].values * sku_norm_cnt_df[
                'store_sku_avg_day_sale'].values
            sku_norm_cnt_df['sku_norm_cnt_upper'] = np.max(
                sku_norm_cnt_df[['sku_norm_cnt_upper', 'sku_norm_cnt_upper_tmp']],
                axis=1
            )
            sku_norm_cnt_df['sku_norm_cnt'] = np.min(
                sku_norm_cnt_df[['sku_norm_cnt', 'sku_norm_cnt_upper']],
                axis=1
            )
            sku_norm_cnt_df.set_index(['store_code', 'cate_level4_code'], inplace=True)

            # 合理周转修正
            # 使用单品组合价对合理周转天数进行调整
            sku_norm_cnt_df['turnover_day'] = store_cate4_turnover_day['turnover_days_avg']
            sku_norm_cnt_df['turnover_day'] = np.where(
                (sku_norm_cnt_df['turnover_day'] == 0) | (sku_norm_cnt_df['turnover_day'].isnull()),
                45,
                sku_norm_cnt_df['turnover_day']
            )
            sku_norm_cnt_df['sku_norm_cnt_turnover_day_fix'] = sku_norm_cnt_df['store_sku_avg_day_sale'].values * \
                                                               sku_norm_cnt_df['turnover_day'].values
            sku_norm_cnt_df['sku_norm_cnt'] = np.amin(
                sku_norm_cnt_df[['sku_norm_cnt', 'sku_norm_cnt_turnover_day_fix']],
                axis=1
            )
            del sku_norm_cnt_df['sku_norm_cnt_turnover_day_fix']
            # 价格修正
            sku_norm_cnt_df['sku_norm_cnt_price_fix'] = np.where(
                (sku_norm_cnt_df['turnover_day'] > 0) & (sku_norm_cnt_df['display_code'] != '2220'),
                np.select(
                    condlist=[
                        sku_norm_cnt_df['combine_price'].between(left=0, right=2, inclusive='left'),
                        sku_norm_cnt_df['combine_price'].between(left=2, right=25, inclusive='left'),
                        sku_norm_cnt_df['combine_price'].between(left=25, right=49, inclusive='left'),
                        sku_norm_cnt_df['combine_price'].between(left=49, right=100, inclusive='left'),
                        # sku_norm_cnt_df['comb_price'].between(left=49, right=100, inclusive='left'),
                    ],
                    choicelist=[25, 10, 6, 3],
                    default=-np.inf
                ),
                -np.inf
            )
            sku_norm_cnt_df['sku_norm_cnt'] = np.max(
                sku_norm_cnt_df[['sku_norm_cnt', 'sku_norm_cnt_price_fix']],
                axis=1
            )
            # 若组合价>=100,那么标准数据=min(合理周转修正，3)
            sku_norm_cnt_df['sku_norm_cnt'] = np.where(
                sku_norm_cnt_df['combine_price'] >= 100,
                np.min(
                    np.vstack((sku_norm_cnt_df['sku_norm_cnt'].values, np.array([3] * len(sku_norm_cnt_df)))),
                    axis=0
                ),
                sku_norm_cnt_df['sku_norm_cnt'].values
            )
            sku_norm_cnt_df['sku_norm_cnt'] = np.round(sku_norm_cnt_df['sku_norm_cnt'] + 1e-8)  # 四舍五入取整
            del sku_norm_cnt_df['sku_norm_cnt_price_fix']

            # 2. 单品初始需求量计算
            sku_init_replenishment_df = sku_norm_cnt_df.copy(deep=True)
            del sku_norm_cnt_df
            # 将需要细类的备货天数放大1.2倍，用于下文计算
            sku_init_replenishment_df['back_up_days'] = np.where(
                sku_init_replenishment_df['cate4_need'] == 1,
                1.2, 1.0) * sku_init_replenishment_df['back_up_days']
            sku_init_replenishment_df['zero_col'] = 0.0
            sku_init_replenishment_df = sku_init_replenishment_df.reset_index().set_index(['store_code', 'goods_code'])
            sku_init_replenishment_df['store_sku_inv'] = store_sku_stock_df['inv']
            sku_init_replenishment_df['store_sku_inv'].fillna(0.0, inplace=True)
            sku_init_replenishment_df['sales_pred_d_avg_store_sku_fixed'] = store_sku_sale_pred_df[
                'sales_pred_d_avg_store_sku_fixed']
            # 用日均销填充预测结果空值
            sku_init_replenishment_df['sales_pred_d_avg_store_sku_fixed_fill_with_avg'] = np.where(
                sku_init_replenishment_df['sales_pred_d_avg_store_sku_fixed'].notnull(),
                sku_init_replenishment_df['sales_pred_d_avg_store_sku_fixed'],
                sku_init_replenishment_df['store_sku_avg_day_sale']
            )
            # 单品初始需求 = 单品标准数量 + 单品预测* 备货天数 - 单品门店库存数
            sku_init_replenishment_df['sku_init_replenishment'] = \
                sku_init_replenishment_df['sku_norm_cnt'].values + \
                sku_init_replenishment_df['sales_pred_d_avg_store_sku_fixed_fill_with_avg'].values * \
                sku_init_replenishment_df['back_up_days'].values - \
                sku_init_replenishment_df['store_sku_inv'].values
            # 大于0处理
            sku_init_replenishment_df['sku_init_replenishment'] = np.max(
                sku_init_replenishment_df[['sku_init_replenishment', 'zero_col']],
                axis=1
            )

            # 不超过：单品标准数量 + 单品预测 * max(备货天数 - 在途天数 ,0)
            sku_init_replenishment_df['sku_init_replenishment_limit'] = \
                sku_init_replenishment_df['sku_norm_cnt'].values + \
                sku_init_replenishment_df['sales_pred_d_avg_store_sku_fixed_fill_with_avg'].values * \
                (sku_init_replenishment_df['back_up_days'].values - sku_init_replenishment_df['intransit_day'].values)

            sku_init_replenishment_df['sku_init_replenishment'] = np.min(
                sku_init_replenishment_df[['sku_init_replenishment', 'sku_init_replenishment_limit']],
                axis=1
            )
            # 规格四舍五入
            sku_init_replenishment_df['sku_init_replenishment'] = np.round(
                sku_init_replenishment_df['sku_init_replenishment'].values / sku_init_replenishment_df[
                    'conversion_qty'].values + 1e-8) * sku_init_replenishment_df['conversion_qty'].values
            # 需求量=0：1.细类定标sku数=0;2.不需要的细类;3.计算结果小于0
            sku_init_replenishment_df['sku_init_replenishment'] = np.where(
                (sku_init_replenishment_df['sku_init_replenishment'] < 0.0) |
                (sku_init_replenishment_df['sku_qty'].values == 0.0) |
                (sku_init_replenishment_df['pcs_qty'].values == 0.0) |
                (sku_init_replenishment_df['cate4_no_need'].values == 1),
                0.0,
                sku_init_replenishment_df['sku_init_replenishment']
            )
            del sku_init_replenishment_df['sku_init_replenishment_limit']
            # 3. 单品初始需求量上限计算
            # 单品标准数量 + 单品综合日均销 * （备货天数 + 发货间隔天数）- 单品门店库存数
            # 日均销大于1.0的，将发货间隔天数调大2倍
            sku_init_replenishment_df['delivery_interval_day'] *= np.where(
                sku_init_replenishment_df['store_sku_week_fix_avg_day_sale'] >= 1.0,
                2,
                1
            )
            # 限制发货间隔*倍数<7
            sku_init_replenishment_df.loc[
                sku_init_replenishment_df['delivery_interval_day'].values > 7, 'delivery_interval_day'] = 7
            sku_init_replenishment_df['sku_init_replenishment_upper'] = \
                sku_init_replenishment_df['sku_norm_cnt'].values + \
                (sku_init_replenishment_df['back_up_days'].values + sku_init_replenishment_df[
                    'delivery_interval_day'].values) * \
                sku_init_replenishment_df['sales_pred_d_avg_store_sku_fixed_fill_with_avg'].values - \
                sku_init_replenishment_df['store_sku_inv'].values
            # 日均销小于0.5的，需求量上限=初始需求量
            sku_init_replenishment_df['sku_init_replenishment_upper'] = np.where(
                sku_init_replenishment_df['store_sku_week_fix_avg_day_sale'] < 0.5,
                sku_init_replenishment_df['sku_init_replenishment'],
                sku_init_replenishment_df['sku_init_replenishment_upper']
            )
            # 结果大于0
            sku_init_replenishment_df['sku_init_replenishment_upper'] = np.max(
                sku_init_replenishment_df[['sku_init_replenishment_upper', 'zero_col']],
                axis=1
            )
            # 规格四舍五入
            sku_init_replenishment_df['sku_init_replenishment_upper'] = np.round(
                sku_init_replenishment_df['sku_init_replenishment_upper'].values / sku_init_replenishment_df[
                    'conversion_qty'].values + 1e-8) * sku_init_replenishment_df['conversion_qty'].values
            # 需求量上限=0：1.细类定标sku数=0;2.不需要的细类;3.结果小于0
            sku_init_replenishment_df['sku_init_replenishment_upper'] = np.where(
                (sku_init_replenishment_df['sku_qty'].values == 0.0) |
                (sku_init_replenishment_df['pcs_qty'].values == 0.0) |
                (sku_init_replenishment_df['cate4_no_need'].values == 1),
                0.0,
                sku_init_replenishment_df['sku_init_replenishment_upper']
            )
            # 新品单独计算初始需求量及上限
            # 初始需求量=初始需求量上限=商品等级&门店等级对应的铺货规格数*中包规格
            sku_init_replenishment_df['new_arrival'] = new_sku_label_df['new_arrival']
            # 如果当前门店库存+在途库存>0,那么强制性判断该品为老品
            sku_init_replenishment_df.loc[sku_init_replenishment_df['store_sku_inv'].values > 0, 'new_arrival'] = 0
            sku_init_replenishment_df_new = sku_init_replenishment_df[sku_init_replenishment_df['new_arrival'] == 1]
            sku_init_replenishment_df_old = sku_init_replenishment_df[sku_init_replenishment_df['new_arrival'] != 1]
            sku_init_replenishment_df_old.set_index('cate_level4_code', append=True, inplace=True)
            if len(sku_init_replenishment_df_new) > 0:
                # 若有新品数据，则拼接规格等数据
                goods_level_df = goods_level_data_load.goods_level()
                cate4_store_level_df = goods_level_data_load.cate4_store_level()
                goods_distribution_specs_df = goods_level_data_load.goods_distribution_specs()
                sku_init_replenishment_df_new = sku_init_replenishment_df_new.reset_index(drop=False).set_index(
                    ['goods_code'])
                sku_init_replenishment_df_new['goods_level'] = goods_level_df['level']
                sku_init_replenishment_df_new = sku_init_replenishment_df_new.reset_index(drop=False).set_index(
                    ['store_code', 'cate_level4_code'])
                sku_init_replenishment_df_new['cate4_store_level'] = cate4_store_level_df['level']
                # 商品等级无效数据剔除，并将其和门店细类等级缺失填充为默认值
                sku_init_replenishment_df_new.loc[
                    ~sku_init_replenishment_df_new['goods_level'].isin(['S', 'A', 'B', 'C']), 'goods_level'] = None
                sku_init_replenishment_df_new.fillna({'goods_level': 'B', 'cate4_store_level': 'C'}, inplace=True)
                sku_init_replenishment_df_new.reset_index(drop=False, inplace=True)
                sku_init_replenishment_df_new = pd.merge(
                    left=sku_init_replenishment_df_new,
                    right=goods_distribution_specs_df,
                    on=['cate_level4_code', 'goods_level', 'cate4_store_level'],
                    how='left'
                )
                sku_init_replenishment_df_new['package_number'].fillna(0.0, inplace=True)
                sku_init_replenishment_df_new['sku_init_replenishment'] = sku_init_replenishment_df_new[
                                                                              'package_number'] * \
                                                                          sku_init_replenishment_df_new[
                                                                              'conversion_qty']
                # 新品首铺属于(0,3)的，直接赋值为3，并用规格向上取整，首铺量>64的，直接赋值为64，并用规格向下取整
                sku_init_replenishment_df_new[
                    sku_init_replenishment_df_new['sku_init_replenishment'].between(0, 3, inclusive='neither')][
                    'sku_init_replenishment'] = 3

                sku_init_replenishment_df_new['sku_init_replenishment'] = np.ceil(
                    sku_init_replenishment_df_new['sku_init_replenishment'] / sku_init_replenishment_df_new[
                        'conversion_qty']
                ) * sku_init_replenishment_df_new['conversion_qty']

                sku_init_replenishment_df_new[
                    sku_init_replenishment_df_new['sku_init_replenishment'] > 64][
                    'sku_init_replenishment'] = 64
                sku_init_replenishment_df_new['sku_init_replenishment'] = np.floor(
                    sku_init_replenishment_df_new['sku_init_replenishment'] / sku_init_replenishment_df_new[
                        'conversion_qty']
                ) * sku_init_replenishment_df_new['conversion_qty']

                # 细类定标=0，则直接取0
                sku_init_replenishment_df_new['sku_init_replenishment'] = np.where(
                    (sku_init_replenishment_df_new['sku_qty'].values == 0) |
                    (sku_init_replenishment_df_new['pcs_qty'].values == 0),
                    0.0,
                    sku_init_replenishment_df_new['sku_init_replenishment']
                )
                sku_init_replenishment_df_new['sku_init_replenishment_upper'] = sku_init_replenishment_df_new[
                    'sku_init_replenishment']
                sku_init_replenishment_df_new.set_index(['store_code', 'goods_code', 'cate_level4_code'], inplace=True)
                self.sku_init_replenishment_df = pd.concat([
                    sku_init_replenishment_df_old[self.output_columns],
                    sku_init_replenishment_df_new[self.output_columns],
                ])
            else:
                self.sku_init_replenishment_df = sku_init_replenishment_df_old[self.output_columns]
            # 女士护肤系列: 基础补货量==基础补货量上限
            self.sku_init_replenishment_df['sku_init_replenishment_upper'] = np.where(
                self.sku_init_replenishment_df.index.get_level_values('cate_level4_code') == '22011656',
                self.sku_init_replenishment_df['sku_init_replenishment'],
                self.sku_init_replenishment_df['sku_init_replenishment_upper']
            )
            self.check_amount(self.sku_init_replenishment_df, len(base_table_df))
            self.check_zero(self.sku_init_replenishment_df, 'sku_init_replenishment')
            self.check_negative(self.sku_init_replenishment_df, 'sku_init_replenishment')
            logger.info("finished SkuInitReplenishment!")

        return self.sku_init_replenishment_df


sku_init_replenishment = SkuInitReplenishment()
