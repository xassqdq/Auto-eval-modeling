# src/utils/file_handler.py
"""
文件读写与路径管理模块
支持 CSV / Excel / JSON 等格式的统一读取与保存接口
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

from .logging_config import get_logger

logger = get_logger("auto_eval.file_handler")

# 支持的数据文件扩展名
SUPPORTED_DATA_FORMATS = {".csv", ".xlsx", ".xls", ".json", ".parquet"}


class FileHandler:
    """
    文件处理器：提供数据文件读写、输出目录管理等功能。

    Attributes
    ----------
    output_root : Path
        输出根目录（默认 ./output）
    """

    def __init__(self, output_root: str | Path = "output"):
        self.output_root = Path(output_root)
        self._ensure_output_dirs()

    # --------------------------------------------------------
    #  目录管理
    # --------------------------------------------------------

    def _ensure_output_dirs(self) -> None:
        """创建标准输出目录结构"""
        subdirs = ["reports", "figures", "scripts", "logs", "intermediate"]
        for sub in subdirs:
            (self.output_root / sub).mkdir(parents=True, exist_ok=True)
        logger.debug(f"输出目录已初始化: {self.output_root.resolve()}")

    def get_output_path(
        self,
        filename: str,
        subdir: str = "",
        add_timestamp: bool = False,
        suffix: str = "",
    ) -> Path:
        """
        构建输出文件路径。

        Parameters
        ----------
        filename : str
            文件名（含扩展名）
        subdir : str
            子目录（如 'figures', 'reports'）
        add_timestamp : bool
            是否在文件名中加入时间戳（防止覆盖）
        suffix : str
            附加后缀（如 '_v2'）

        Returns
        -------
        Path
        """
        stem = Path(filename).stem
        ext = Path(filename).suffix

        if add_timestamp:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{stem}_{ts}{suffix}{ext}"
        elif suffix:
            filename = f"{stem}{suffix}{ext}"

        target_dir = self.output_root / subdir if subdir else self.output_root
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir / filename

    def clean_output(self, subdir: str = "", confirm: bool = False) -> None:
        """
        清理输出目录。

        Parameters
        ----------
        subdir : str
            若指定，只清理该子目录；否则清理整个输出根目录
        confirm : bool
            安全开关，必须为 True 才会执行删除
        """
        if not confirm:
            logger.warning("clean_output 需要 confirm=True 才会执行，操作已跳过")
            return

        target = self.output_root / subdir if subdir else self.output_root
        if target.exists():
            shutil.rmtree(target)
            logger.info(f"已清理输出目录: {target}")
        self._ensure_output_dirs()

    # --------------------------------------------------------
    #  数据文件读取
    # --------------------------------------------------------

    def read_data(
        self,
        file_path: str | Path,
        sheet_name: Union[str, int] = 0,
        index_col: Optional[str] = None,
        encoding: str = "utf-8",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        统一读取数据文件接口，自动根据扩展名选择读取方式。

        Parameters
        ----------
        file_path : str | Path
            数据文件路径
        sheet_name : str | int
            Excel 工作表名称或索引（仅 Excel 生效）
        index_col : str | None
            作为行索引的列名
        encoding : str
            CSV 文件编码
        **kwargs :
            传递给 pandas 读取函数的额外参数

        Returns
        -------
        pd.DataFrame
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {file_path}")

        suffix = file_path.suffix.lower()
        if suffix not in SUPPORTED_DATA_FORMATS:
            raise ValueError(
                f"不支持的文件格式: '{suffix}'。"
                f"支持格式: {SUPPORTED_DATA_FORMATS}"
            )

        logger.info(f"正在读取数据文件: {file_path}")

        try:
            if suffix == ".csv":
                df = pd.read_csv(
                    file_path,
                    encoding=encoding,
                    index_col=index_col,
                    **kwargs,
                )
            elif suffix in {".xlsx", ".xls"}:
                df = pd.read_excel(
                    file_path,
                    sheet_name=sheet_name,
                    index_col=index_col,
                    **kwargs,
                )
            elif suffix == ".json":
                df = pd.read_json(file_path, encoding=encoding, **kwargs)
                if index_col and index_col in df.columns:
                    df = df.set_index(index_col)
            elif suffix == ".parquet":
                df = pd.read_parquet(file_path, **kwargs)
                if index_col and index_col in df.columns:
                    df = df.set_index(index_col)
            else:
                raise ValueError(f"未处理的格式: {suffix}")

        except Exception as e:
            logger.error(f"读取文件失败: {e}")
            raise

        logger.info(
            f"数据读取成功: shape={df.shape}, "
            f"columns={list(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}"
        )
        return df

    # --------------------------------------------------------
    #  数据文件保存
    # --------------------------------------------------------

    def save_dataframe(
        self,
        df: pd.DataFrame,
        filename: str,
        subdir: str = "intermediate",
        fmt: str = "csv",
        add_timestamp: bool = False,
        **kwargs: Any,
    ) -> Path:
        """
        保存 DataFrame 到文件。

        Parameters
        ----------
        df : pd.DataFrame
        filename : str
            输出文件名（不含扩展名）
        subdir : str
            输出子目录
        fmt : str
            输出格式：'csv', 'xlsx', 'json', 'parquet'
        add_timestamp : bool
            是否加时间戳

        Returns
        -------
        Path
            实际保存路径
        """
        ext_map = {"csv": ".csv", "xlsx": ".xlsx", "json": ".json", "parquet": ".parquet"}
        ext = ext_map.get(fmt, ".csv")
        out_path = self.get_output_path(
            filename + ext, subdir=subdir, add_timestamp=add_timestamp
        )

        if fmt == "csv":
            df.to_csv(out_path, encoding="utf-8-sig", **kwargs)  # utf-8-sig 支持 Excel 打开
        elif fmt == "xlsx":
            df.to_excel(out_path, **kwargs)
        elif fmt == "json":
            df.to_json(out_path, force_ascii=False, indent=2, **kwargs)
        elif fmt == "parquet":
            df.to_parquet(out_path, **kwargs)
        else:
            raise ValueError(f"不支持的保存格式: {fmt}")

        logger.info(f"DataFrame 已保存: {out_path} ({fmt})")
        return out_path

    def save_dict(
        self,
        data: dict[str, Any],
        filename: str,
        subdir: str = "intermediate",
        add_timestamp: bool = False,
    ) -> Path:
        """保存字典为 JSON 文件"""
        out_path = self.get_output_path(
            filename + ".json", subdir=subdir, add_timestamp=add_timestamp
        )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"字典已保存: {out_path}")
        return out_path

    def save_text(
        self,
        content: str,
        filename: str,
        subdir: str = "reports",
        add_timestamp: bool = False,
    ) -> Path:
        """保存文本内容（如 LaTeX 源码）到文件"""
        out_path = self.get_output_path(
            filename, subdir=subdir, add_timestamp=add_timestamp
        )
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"文本文件已保存: {out_path}")
        return out_path

    # --------------------------------------------------------
    #  工具方法
    # --------------------------------------------------------

    @staticmethod
    def detect_encoding(file_path: str | Path) -> str:
        """
        自动检测文件编码（需安装 chardet）。
        若 chardet 未安装，默认返回 'utf-8'。
        """
        try:
            import chardet
            with open(file_path, "rb") as f:
                raw = f.read(min(32768, Path(file_path).stat().st_size))
            result = chardet.detect(raw)
            encoding = result.get("encoding", "utf-8") or "utf-8"
            logger.debug(f"检测到文件编码: {encoding} (置信度: {result.get('confidence', 0):.2f})")
            return encoding
        except ImportError:
            return "utf-8"

    @staticmethod
    def list_files(
        directory: str | Path,
        extensions: Optional[set[str]] = None,
        recursive: bool = False,
    ) -> list[Path]:
        """
        列出目录下的文件。

        Parameters
        ----------
        directory : str | Path
        extensions : set[str] | None
            文件扩展名过滤，如 {'.csv', '.xlsx'}
        recursive : bool
            是否递归子目录

        Returns
        -------
        list[Path]
        """
        directory = Path(directory)
        if not directory.exists():
            return []

        pattern = "**/*" if recursive else "*"
        files = [p for p in directory.glob(pattern) if p.is_file()]

        if extensions:
            files = [p for p in files if p.suffix.lower() in extensions]

        return sorted(files)

    def copy_to_output(
        self,
        src: str | Path,
        subdir: str = "",
        rename: Optional[str] = None,
    ) -> Path:
        """将外部文件复制到输出目录"""
        src = Path(src)
        filename = rename if rename else src.name
        dst = self.get_output_path(filename, subdir=subdir)
        shutil.copy2(src, dst)
        logger.info(f"文件已复制: {src} → {dst}")
        return dst