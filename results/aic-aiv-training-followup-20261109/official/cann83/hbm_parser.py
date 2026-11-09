#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2018-2019. All rights reserved.

import logging
import os
import struct

from common_func.constant import Constant
from common_func.db_name_constant import DBNameConstant
from common_func.file_manager import FileManager
from common_func.file_manager import FileOpen
from common_func.info_conf_reader import InfoConfReader
from common_func.ms_constant.number_constant import NumberConstant
from common_func.ms_multi_process import MsMultiProcess
from common_func.msprof_exception import ProfException
from common_func.msvp_common import is_valid_original_data
from common_func.path_manager import PathManager
from framework.offset_calculator import OffsetCalculator
from msmodel.hardware.hbm_model import HbmModel
from msparser.data_struct_size_constant import StructFmt
from profiling_bean.prof_enum.data_tag import DataTag


class ParsingHBMData(MsMultiProcess):
    """
    parsing DDR data class
    """

    def __init__(self: any, file_list: dict, sample_config: dict) -> None:
        super().__init__(sample_config)
        self.hbm_data = []
        self.sample_config = sample_config
        self._file_list = file_list.get(DataTag.HBM, [])
        self.project_path = sample_config.get("result_dir", "")
        self.calculate = OffsetCalculator(self._file_list, StructFmt.HBM_FMT_SIZE, self.project_path)
        self._model = HbmModel(self.project_path, DBNameConstant.DB_HBM,
                               [DBNameConstant.TABLE_HBM_ORIGIN, DBNameConstant.TABLE_HBM_BW])
        self._file_list.sort(key=lambda x: int(x.split("_")[-1]))

    @staticmethod
    def _update_hbm_data(start_time: int, item: list, headers: list) -> tuple:
        item[0] = start_time + item[0] * NumberConstant.USTONS
        item[2] = '{}'.format('read' if item[2] == 0 else 'write')
        return tuple(headers + item)

    def read_binary_data(self: any, file_name: str, device_id: str, replay_id: str) -> int:
        """
        parsing hbm data and insert into hbm.db
        """
        status = NumberConstant.ERROR
        hbm_file = PathManager.get_data_file_path(self.project_path, file_name)
        if not os.path.exists(hbm_file):
            return status
        _file_size = os.path.getsize(hbm_file)
        try:
            with FileOpen(hbm_file, "rb") as hbm_f:
                hbm_data = self.calculate.pre_process(hbm_f.file_reader, _file_size)
                struct_nums = _file_size // StructFmt.HBM_FMT_SIZE
                struct_data = struct.unpack(StructFmt.BYTE_ORDER_CHAR + StructFmt.HBM_FMT * struct_nums,
                                            hbm_data)
                start_time = InfoConfReader().get_start_timestamp()
                for i in range(struct_nums):
                    self.hbm_data.append(self._update_hbm_data(start_time, list(struct_data[i * 4:(i + 1) * 4]),
                                                               [device_id, replay_id]))
            return NumberConstant.SUCCESS
        except (OSError, SystemError, ValueError, TypeError, RuntimeError) as err:
            logging.error("%s: %s", file_name, err, exc_info=Constant.TRACE_BACK_SWITCH)
            return status

    def start_parsing_data_file(self: any) -> None:
        """
        parsing data file
        """
        try:
            for file_name in self._file_list:
                if is_valid_original_data(file_name, self.project_path):
                    self._original_data_handler(file_name)
        except (OSError, SystemError, ValueError, TypeError, RuntimeError) as err:
            logging.error(str(err), exc_info=Constant.TRACE_BACK_SWITCH)

    def save(self: any) -> None:
        """
        save data to db
        :return: None
        """
        if self.hbm_data and self._model:
            self._model.init()
            self._model.create_table()
            self._model.flush(self.hbm_data)
            self._model.insert_bw_data(self.sample_config.get("hbm_profiling_events", "").split(","))
            self._model.finalize()

    def ms_run(self: any) -> None:
        """
        main
        :return: None
        """
        try:
            if self._file_list:
                self.start_parsing_data_file()
                self.save()
        except (OSError, SystemError, ValueError, TypeError, RuntimeError, ProfException) as hbm_err:
            logging.error(str(hbm_err), exc_info=Constant.TRACE_BACK_SWITCH)

    def _original_data_handler(self: any, file_name: str) -> None:
        device_id = self.sample_config.get("device_id", "0")
        logging.info("start parsing HBM data file: %s", file_name)
        status = self.read_binary_data(file_name, device_id, '0')  # replay is is 0
        FileManager.add_complete_file(self.project_path, file_name)
        if status:
            logging.error('Insert HBM bandwidth data error.')
        logging.info("Create HBM DB finished!")