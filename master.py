# -*- encoding: utf-8 -*-
"""
@Time        : 2022/2/23 3:03 PM
@Author      : miniso
@Description : 主程序
"""
import numpy as np
import pandas as pd
from central_replenishment import central_replenishment
from common import CommonObject
from data_load import SaleDataLoad, Cate4Demand
from data_load import mission_info_data_load
from data_load import stock_data_load
from ship_money_reduce import ship_money_reduce
from loguru import logger
from utils.tools import common_output


class Master(CommonObject):
    def __init__(self):
        super().__init__()
        self.cal_result_df = None
        self.database_connector = CommonObject.config['database_connector']
        self.detail_result_output_table = CommonObject.config['detail_result_output_table']
        self.stat_result_output_table = CommonObject.config['stat_result_output_table']

    @logger.catch
    def calculate(self) -> pd.DataFrame:
        if self.cal_result_df is None:
            # 常规补货计算流程
            if self.args.replenish_type == 'normal':
                self.cal_result_df = self._normal_calculate()
            # 统配补货计算流程
            elif self.args.replenish_type == 'centralized':
                self.cal_result_df = self._centralized_calculate()
            else:
                logger.error(f"You choose a replenish_type :{self.args.replenish_type},this is not supported now!")
                raise
        return self.cal_result_df

    @logger.catch
    def _normal_calculate(self) -> pd.DataFrame:
        logger.info("start run normal replenish!")
        output_columns = ['replenish_qty_suggest', 'suggest_reason', 'store_sku_inv',
                          'sales_pred_d_avg_store_sku_fixed',
                          'sales_pred_d_avg_store_cate4_fixed', 'cate_level4_code',
                          'sku_norm_cnt', 'mainwarehouse']
        # 如果建议单全部是禁配商品，则直接返回一个空结果
        if len(mission_info_data_load.mission_base_info_df) == 0:
            cal_result_df = pd.DataFrame(
                columns=['store_code', 'goods_code'] + output_columns + ['is_suggest', 'suggest_reason_code']
            )
            cal_result_df.set_index(['store_code', 'goods_code'], inplace=True)
        else:
            ware_alloc_df = ship_money_reduce.calculate()
            cal_result_df = ware_alloc_df.copy(deep=True)
            cal_result_df.rename(
                columns={'alloc_cnt_money_reduce': 'replenish_qty_suggest',
                         'alloc_reason_money_reduce': 'suggest_reason'},
                inplace=True
            )
            cal_result_df['replenish_qty_suggest'].fillna(0.0, inplace=True)
            cal_result_df['is_suggest'] = np.where(cal_result_df['replenish_qty_suggest'] > 0.0, 1, 0)
            # 保底是2,常规是3
            cal_result_df['suggest_reason_code'] = np.where(
                cal_result_df['sku_minimum'] > 0, 2, 3
            )
            del cal_result_df['sku_minimum']

        # 禁配数据整理
        store_sku_forbidden_df = mission_info_data_load.mission_base_info_df_forbid[
            ['store_code', 'goods_code', 'cate_level4_code', 'is_suggest', 'suggest_reason_code',
             'mainwarehouse']].set_index(
            ['store_code', 'goods_code'])
        # 如果有禁配数据，则进行禁配数据拼接操作
        if len(store_sku_forbidden_df) > 0:
            store_sku_forbidden_df['suggest_reason_code'] = store_sku_forbidden_df['suggest_reason_code'].astype(
                int)
            # 销量预测
            store_sku_sale_pred_df = mission_info_data_load.store_sku_sale_pred()[
                ['sales_pred_d_avg_store_sku_fixed']]
            store_sku_forbidden_df['sales_pred_d_avg_store_sku_fixed'] = store_sku_sale_pred_df[
                'sales_pred_d_avg_store_sku_fixed']
            # 门店库存
            store_sku_stock_df = stock_data_load.store_sku_stock()[['inv']]
            store_sku_forbidden_df['store_sku_inv'] = store_sku_stock_df['inv']
            store_sku_forbidden_df.set_index('cate_level4_code', append=True, inplace=True)
            # 门店细类预测
            store_cate4_sale_pred_df = mission_info_data_load.store_cate4_sale_pred()[
                ['sales_pred_d_avg_store_cate4_fixed']]
            store_sku_forbidden_df = pd.merge(left=store_sku_forbidden_df,
                                              right=store_cate4_sale_pred_df,
                                              left_index=True,
                                              right_index=True,
                                              how='left'
                                              )
            store_sku_forbidden_df.reset_index('cate_level4_code', drop=False, inplace=True)
            store_sku_forbidden_df['is_suggest'] = 0
            store_sku_forbidden_df['replenish_qty_suggest'] = 0.0
            store_sku_forbidden_df['suggest_reason'] = None
            store_sku_forbidden_df['sku_norm_cnt'] = 0.0
            cal_result_df = pd.concat([
                cal_result_df[output_columns + ['is_suggest', 'suggest_reason_code']],
                store_sku_forbidden_df[output_columns + ['is_suggest', 'suggest_reason_code']]], axis=0,
                ignore_index=False)
        cal_result_df.set_index('cate_level4_code', append=True, inplace=True)
        cal_result_df.rename(
            columns={
                'sales_pred_d_avg_store_cate4_fixed': 'sales_pred_qty_store_cate4',
                'sales_pred_d_avg_store_sku_fixed': 'sales_pred_qty_store_sku'},
            inplace=True
        )
        cal_result_df['replenish_qty_sku'] = cal_result_df['replenish_qty_suggest']
        cal_result_df.fillna({'sku_norm_cnt': 0.0, 'store_sku_inv': 0.0}, inplace=True)
        cal_result_df['inv_qty_after_replenishment_sku'] = cal_result_df['replenish_qty_sku'] + cal_result_df[
            'store_sku_inv']
        cal_result_df['inv_qty_before_replenishment_sku'] = cal_result_df['store_sku_inv']
        cal_result_df['inv_qty_before_replenishment_cate4'] = cal_result_df.groupby(
            level=['store_code', 'cate_level4_code'])['store_sku_inv'].transform('sum')
        cal_result_df['replenish_qty_cate4'] = cal_result_df.groupby(level=['store_code', 'cate_level4_code'])[
            'replenish_qty_suggest'].transform('sum')
        cal_result_df['inv_qty_after_replenishment_cate4'] = \
            cal_result_df.groupby(level=['store_code', 'cate_level4_code'])[
                'inv_qty_after_replenishment_sku'].transform('sum')
        # 补前sku数
        cal_result_df['goods_code_tmp0'] = np.where(
            cal_result_df['store_sku_inv'] > 0.0,
            cal_result_df.index.get_level_values('goods_code'),
            None
        )
        cal_result_df['inv_sku_species_before_replenishment_cate4'] = cal_result_df.groupby(
            level=['store_code', 'cate_level4_code'])['goods_code_tmp0'].transform('nunique')
        # 计算sku数
        cal_result_df['goods_code_tmp1'] = np.where(
            cal_result_df['replenish_qty_suggest'] > 0.0,
            cal_result_df.index.get_level_values('goods_code'),
            None
        )
        cal_result_df['replenish_sku_species_cate4'] = cal_result_df.groupby(
            level=['store_code', 'cate_level4_code'])['goods_code_tmp1'].transform('nunique')
        cal_result_df['goods_code_tmp2'] = np.where(
            cal_result_df['inv_qty_after_replenishment_sku'] > 0.0,
            cal_result_df.index.get_level_values('goods_code'),
            None
        )
        cal_result_df['inv_sku_species_after_replenishment_cate4'] = cal_result_df.groupby(
            level=['store_code', 'cate_level4_code'])['goods_code_tmp2'].transform('nunique')
        del cal_result_df['goods_code_tmp1'], cal_result_df['goods_code_tmp2']
        cal_result_df['tiny_category_code'] = cal_result_df.index.get_level_values('cate_level4_code')
        # 计算补后仓库库存
        cal_result_df = cal_result_df.reset_index().set_index(['mainwarehouse', 'goods_code'])
        warehouse_sku_stock_df = stock_data_load.warehouse_sku_stock()[['inv_available']]  # 仓库商品库存读取
        cal_result_df['inv_qty_before_replenishment_wrh_sku'] = warehouse_sku_stock_df['inv_available']
        cal_result_df['replenish_qty_wrh_sku'] = cal_result_df.groupby(level=['mainwarehouse', 'goods_code'])[
            'replenish_qty_sku'].transform('sum')
        cal_result_df['inv_qty_after_replenishment_wrh_sku'] = cal_result_df['inv_qty_before_replenishment_wrh_sku'] - \
                                                               cal_result_df['replenish_qty_wrh_sku']
        cal_result_df = cal_result_df.reset_index().set_index(['store_code', 'goods_code'])
        # 删除非必要的字段
        del cal_result_df['mainwarehouse'], cal_result_df['cate_level4_code']
        if len(cal_result_df) != self.args.batch_size:
            logger.error(
                f"master calculate error:data count is {len(cal_result_df)}"
                f",not equal to base table count {self.args.batch_size}"
            )
        logger.info(f"finished normal replenish task calculate!")
        return cal_result_df

    @logger.catch
    def _centralized_calculate(self) -> pd.DataFrame:
        # 统配计算逻辑
        output_columns = ['replenish_qty_suggest', 'is_suggest', 'suggest_reason_code', 'suggest_reason']
        cal_result_df = central_replenishment.calculate()[output_columns]
        # 禁配
        store_sku_forbidden_df = mission_info_data_load.mission_base_info_df_forbid[
            ['store_code', 'goods_code']].set_index(['store_code', 'goods_code'])
        # 如果有禁配数据
        if len(store_sku_forbidden_df) > 0:
            store_sku_forbidden_df['replenish_qty_suggest'] = 0
            store_sku_forbidden_df['is_suggest'] = 0
            store_sku_forbidden_df['suggest_reason_code'] = '1'
            store_sku_forbidden_df['suggest_reason'] = '禁配'
            cal_result_df = pd.concat([
                cal_result_df[output_columns],
                store_sku_forbidden_df[output_columns]
            ])

        base = mission_info_data_load.mission_base_info_df_org.reset_index()[
            ['task_batch_code', 'suggest_code', 'suggest_detail_code', 'mainwarehouse', 'store_code',
             'goods_code']].set_index(['store_code', 'goods_code'])

        if len(cal_result_df) != len(base):
            logger.error(
                f"CentralReplenishment calculate error:all data add forbidden count is  {len(cal_result_df)}"
                f",not equal to base table count {len(base)}"
            )

        cal_result_df = pd.merge(left=base,
                                 right=cal_result_df,
                                 left_index=True,
                                 right_index=True
                                 )
        cal_result_df.reset_index(drop=False, inplace=True)
        cal_result_df.rename(
            columns={'mainwarehouse': 'supplier_code', 'store_code': 'demander_code', 'goods_code': 'sku_code'},
            inplace=True
        )
        return cal_result_df

    @logger.catch
    def result_output(self):
        # 常规补货结果输出
        if self.args.replenish_type == 'normal':
            self._normal_result_output()
        # 统配补货结果输出
        elif self.args.replenish_type == 'centralized':
            self._centralized_result_output()
        else:
            logger.error(f"You choose a  replenish_type :{self.args.replenish_type},this is not supported now!")
            raise
        logger.info(f"task run successfully!")

    @logger.catch
    def _normal_result_output(self):
        base_df = mission_info_data_load.mission_base_info_df_org.reset_index().set_index(['store_code', 'goods_code'])
        del base_df['is_suggest'], base_df['suggest_reason_code']
        result_df = pd.merge(base_df, self.cal_result_df, left_index=True, right_index=True)
        result_df = pd.merge(result_df, mission_info_data_load.label_info(), left_index=True, right_index=True,
                             how='left')
        result_df['sku_label'] = result_df['top_sales'].fillna(0).astype(int) * 2 + result_df['new_arrival'].fillna(
            0).astype(int)
        result_df['cate4_label'] = 0
        result_df.reset_index(inplace=True)
        result_df.rename(
            columns={'store_code': 'demander_code', 'goods_code': 'sku_code', 'mainwarehouse': 'supplier_code'},
            inplace=True
        )
        result_df['inv_display_meters_after_replenish_cate4'] = np.where(
            result_df.per_meter_display_qty > 0,
            np.round(result_df.inv_qty_after_replenishment_cate4 / result_df.per_meter_display_qty, 3),
            0)
        # 字段格式修正
        for col in ['replenish_qty_suggest', 'sku_norm_cnt', 'sales_pred_qty_store_sku', 'replenish_qty_sku',
                    'inv_qty_after_replenishment_sku', 'sales_pred_qty_store_cate4', 'replenish_qty_cate4',
                    'per_meter_display_qty', 'inv_qty_before_replenishment_cate4', 'inv_qty_before_replenishment_sku']:
            result_df[col] = result_df[col].astype('float32')
        if len(result_df) != self.args.batch_size:
            logger.error(
                f"master result_output error:data count is {len(result_df)}"
                f",not equal to base table count {self.args.batch_size}"
            )

        result_df['sku_replenished'] = np.where(result_df.replenish_qty_suggest > 0.0, result_df.sku_code, None)
        result_df['sku_available'] = np.where(result_df.inv_qty_after_replenishment_sku > 0.0, result_df.sku_code, None)

        result_df['inv_qty_after_replenishment_new_arrival'] = np.where(result_df.new_arrival.notnull(),
                                                                        result_df.inv_qty_after_replenishment_sku,
                                                                        None)
        result_df['new_sku_available'] = np.where(result_df.inv_qty_after_replenishment_sku > 0.0,
                                                  result_df.new_arrival, None)
        result_df['inv_qty_after_replenishment_top_sales'] = np.where(result_df.top_sales.notnull(),
                                                                      result_df.inv_qty_after_replenishment_sku,
                                                                      None)
        result_df['top_sku_available'] = np.where(result_df.inv_qty_after_replenishment_sku > 0.0, result_df.top_sales,
                                                  None)

        result_df['inv_amt_before_replenishment'] = result_df.store_sku_inv * result_df.combine_price
        result_df['replenish_amt'] = result_df.replenish_qty_suggest * result_df.combine_price

        result_df[
            'inv_amt_after_replenishment_sku'] = result_df.inv_qty_after_replenishment_sku * result_df.combine_price
        result_df['sku_available_before_replenishment'] = np.where(result_df.store_sku_inv > 0.0,
                                                                   result_df.sku_code, None)
        stat_result_df = result_df.groupby(['task_batch_code', 'demander_code'], as_index=False). \
            agg(inv_qty_before_replenishment=('store_sku_inv', 'sum'),
                inv_amt_before_replenishment=('inv_amt_before_replenishment', 'sum'),
                replenish_qty=('replenish_qty_suggest', 'sum'),
                replenish_amt=('replenish_amt', 'sum'),
                inv_qty_after_replenishment=('inv_qty_after_replenishment_sku', 'sum'),
                inv_amt_after_replenishment=('inv_amt_after_replenishment_sku', 'sum'),
                inv_sku_species_before_replenishment=('sku_available_before_replenishment', 'nunique'),
                replenish_sku_species=('sku_replenished', 'nunique'),
                inv_sku_species_after_replenishment=('sku_available', 'nunique'),
                inv_qty_after_replenishment_new_arrival=('inv_qty_after_replenishment_new_arrival', 'sum'),  # 新品
                inv_sku_species_after_replenishment_new_arrival=('new_sku_available', 'sum'),
                inv_qty_after_replenishment_top_sku=('inv_qty_after_replenishment_top_sales', 'sum'),
                inv_sku_species_after_replenishment_top_sku=('top_sku_available', 'sum')
                # 畅销品
                )
        stock_batch_code = stock_data_load.store_sku_stock()['batch_code'][0]
        stat_result_df['inv_batch_code'] = stock_batch_code
        stat_result_df['inv_amt_before_replenishment'] = np.round(
            stat_result_df['inv_amt_before_replenishment'].astype('float'), 2)

        stat_result_df['replenish_amt'] = np.round(
            stat_result_df['replenish_amt'].astype('float'), 2)

        stat_result_df['inv_amt_after_replenishment'] = np.round(
            stat_result_df['inv_amt_after_replenishment'].astype('float'),
            2
        )

        common_output(df=stat_result_df,
                      url=self.config.get('inform_url'),
                      table_name=self.stat_result_output_table,
                      conn=self.database_connector,
                      )

        common_output(df=result_df,
                      url=self.config.get('inform_url'),
                      table_name=self.detail_result_output_table,
                      conn=self.database_connector,
                      )

    @logger.catch
    def _centralized_result_output(self):
        common_output(df=self.cal_result_df,
                      url=self.config.get('inform_url'),
                      table_name=self.detail_result_output_table,
                      conn=self.database_connector,
                      )

    @logger.catch
    def data_prepare(self):
        sale_data_load = SaleDataLoad()
        cate4_demand_load = Cate4Demand()
        sale_data_load.store_sku_sale()
        sale_data_load.store_cate4_sale()
        cate4_demand_load.cate4_need()
        cate4_demand_load.cate4_no_need()
