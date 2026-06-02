from app.skills.base import BaseSkill


class ChatSkill(BaseSkill):
    name = "闲聊工具"
    description = "仅用于处理用户的问候、感谢等闲聊类请求，不涉及任何业务分析"

    def run(self, query: str) -> str:
        if any(w in query for w in ["你好", "哈喽", "嗨", "早上好", "下午好", "晚上好"]):
            return "你好呀！我是B站运营智能助手，能帮你优化标题、预测播放量，有任何问题都可以问～"
        if any(w in query for w in ["谢谢", "感谢", "多谢", "辛苦了"]):
            return "不客气啦！能帮到你我超开心，有其他问题随时都可以问～"
        return "哈哈，你是不是想和我闲聊呀？尽管说～"
