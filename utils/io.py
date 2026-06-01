"""线程安全的增量结果写入器。

把每条样本以 ``img_path`` 为主键写入同一个 jsonl，
在并发 worker 提交结果时按 ``save_interval`` 周期性落盘，崩溃也不会丢全量进度。
"""

from __future__ import annotations

import json
import os
import traceback
from threading import Lock

from .signal_utils import install_signal_handlers_once


class ResultWriter:
    """周期性落盘 + 全量替换写入；失败回退到 .tmp 文件。"""

    def __init__(self, output_file: str, processed: dict[str, dict], save_interval: int = 1):
        self.output_file = output_file
        self.processed = processed
        self.lock = Lock()
        self.tmp_file = output_file + ".tmp"
        self.save_interval = save_interval
        self.update_count = 0
        self.last_save_count = 0
        install_signal_handlers_once()

    def update_and_save(self, result: dict, force_save: bool = False) -> None:
        with self.lock:
            key = result.get("img_path", "")
            if not key:
                return
            self.processed[key] = result
            self.update_count += 1
            if force_save or (self.update_count - self.last_save_count >= self.save_interval):
                self._save_to_disk()
                self.last_save_count = self.update_count

    def _save_to_disk(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.output_file) or ".", exist_ok=True)
            with open(self.tmp_file, "w", encoding="utf-8") as f:
                for data in self.processed.values():
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")
            if os.path.exists(self.output_file):
                os.remove(self.output_file)
            os.rename(self.tmp_file, self.output_file)
        except Exception as e:
            print(f"保存到磁盘时出错: {e}")
            traceback.print_exc()

    def finalize(self) -> None:
        with self.lock:
            try:
                self._save_to_disk()
            except Exception as e:
                print(f"保存最终结果时出错: {e}")
                traceback.print_exc()
            finally:
                if os.path.exists(self.tmp_file):
                    try:
                        os.remove(self.tmp_file)
                    except Exception:
                        pass
