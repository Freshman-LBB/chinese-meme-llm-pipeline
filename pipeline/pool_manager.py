import json
import os
import random


class PoolManager:
    def __init__(self, pool_file=None):
        if pool_file is None:
            pool_file = os.path.join(os.path.dirname(__file__), "pools", "Pool.json")
        self.pool_file = pool_file
        self.pool_data = self._load_pool()

    def _load_pool(self):
        try:
            with open(self.pool_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError:
            print("Pool文件格式错误，请检查")
            return {}

    def get_attitude(self, theme, harm_flag):
        """获取二级态度池结果"""
        try:
            attitudes = self.pool_data.get(theme, {}).get('attitude', {}).get(str(harm_flag), [])
            return random.choice(attitudes) if attitudes else None
        except Exception as e:
            print(f"获取态度池出错: {e}")
            return None

    def get_scene(self, theme, harm_flag):
        """获取三级场景池结果"""
        try:
            scenes = self.pool_data.get(theme, {}).get('scene', {}).get(str(harm_flag), [])
            return random.choice(scenes) if scenes else None
        except Exception as e:
            print(f"获取场景池出错: {e}")
            return None

    def get_style(self, theme, harm_flag):
        """获取风格池结果"""
        try:
            styles = self.pool_data.get(theme, {}).get('style', {}).get(str(harm_flag), [])
            return random.choice(styles) if styles else None
        except Exception as e:
            print(f"获取风格池出错: {e}")
            return None

    def add_pool_item(self, theme, pool_type, harm_flag, item):
        """动态添加池子元素"""
        if theme not in self.pool_data:
            self.pool_data[theme] = {}

        if pool_type not in self.pool_data[theme]:
            self.pool_data[theme][pool_type] = {}

        if str(harm_flag) not in self.pool_data[theme][pool_type]:
            self.pool_data[theme][pool_type][str(harm_flag)] = []

        self.pool_data[theme][pool_type][str(harm_flag)].append(item)
        self._save_pool()

    def _save_pool(self):
        """保存池子数据"""
        with open(self.pool_file, 'w', encoding='utf-8') as f:
            json.dump(self.pool_data, f, ensure_ascii=False, indent=2)

    def delete_pool_item(self, theme, pool_type, harm_flag, item):
        """删除池子元素"""
        try:
            self.pool_data[theme][pool_type][str(harm_flag)].remove(item)
            self._save_pool()
        except ValueError:
            print(f"未找到对应元素：{item}")
        except KeyError:
            print(f"未找到对应主题或池子类型：{theme}, {pool_type}")