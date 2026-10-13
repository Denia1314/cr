"""Offline illustrated choices for physical battlefield annotation."""
import base64
import io
from pathlib import Path
from PIL import Image
from .cards import CardCatalog

ZH=dict(zip(
    'knight archers goblins giant pekka minions balloon witch barbarians golem skeletons valkyrie skeleton_army bomber musketeer baby_dragon prince wizard mini_pekka spear_goblins giant_skeleton hog_rider minion_horde ice_wizard royal_giant guards princess dark_prince three_musketeers lava_hound ice_spirit fire_spirit miner sparky bowler lumberjack battle_ram inferno_dragon ice_golem mega_minion dart_goblin goblin_gang electro_wizard elite_barbarians hunter executioner bandit royal_recruits night_witch bats royal_ghost ram_rider zappies rascals cannon_cart mega_knight skeleton_barrel flying_machine wall_breakers royal_hogs goblin_giant fisherman magic_archer electro_dragon firecracker mighty_miner elixir_golem battle_healer skeleton_king archer_queen golden_knight monk skeleton_dragons mother_witch electro_spirit electro_giant phoenix cannon goblin_hut mortar inferno_tower bomb_tower barbarian_hut tesla elixir_collector x_bow tombstone furnace goblin_cage goblin_drill heal_spirit little_prince goblin_demolisher goblin_machine suspicious_bush goblinstein rune_giant berserker boss_bandit'.split(),
    '骑士 弓箭手 哥布林 巨人 皮卡超人 亡灵 气球兵 女巫 野蛮人 戈仑石人 骷髅兵 瓦基丽武神 骷髅军团 炸弹兵 火枪手 宝宝龙 王子 法师 迷你皮卡 哥布林投矛手 骷髅巨人 野猪骑士 亡灵大军 寒冰法师 皇家巨人 骷髅守卫 公主 黑暗王子 三个火枪手 熔岩猎犬 冰雪精灵 烈焰精灵 掘地矿工 电磁炮 巨石投手 狂暴樵夫 野蛮人攻城槌 地狱飞龙 戈仑冰人 重甲亡灵 吹箭哥布林 哥布林团伙 闪电法师 精锐野蛮人 猎人 飞斧屠夫 幻影刺客 皇家卫队 暗夜女巫 蝙蝠 皇家幽灵 蛮羊骑士 电击车小队 绿林团伙 加农炮战车 超级骑士 骷髅气球 飞行器 炸弹人 皇家野猪 哥布林巨人 渔夫 神箭游侠 雷电飞龙 烟花炮手 威猛矿工 圣水戈仑 战斗天使 骷髅帝王 弓箭女皇 黄金圣骑士 武僧 骷髅飞龙 野猪女巫 雷电精灵 雷电巨人 凤凰 加农炮 哥布林小屋 迫击炮 地狱之塔 炸弹塔 野蛮人小屋 特斯拉电磁塔 圣水收集器 X连弩 骷髅墓碑 烈焰熔炉 哥布林牢笼 哥布林钻机 治疗精灵 小王子 哥布林爆破手 哥布林机甲 可疑草丛 哥布林科学怪人 符文巨人 狂战士 头领刺客'.split()))


def illustrated_cards():
    catalog=CardCatalog.load(Path(__file__).resolve().parents[1]/'data/cards.json')
    rows=[]
    for card in catalog.cards:
        if card.kind not in ('troop','building'):continue
        picture='';path=catalog.local_icon_path(card)
        if path and path.is_file():
            with Image.open(path) as source:
                source=source.convert('RGBA');source.thumbnail((96,112));buffer=io.BytesIO();source.save(buffer,format='PNG')
            picture='data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()
        rows.append(dict(id=card.card_id,name=card.name_zh or ZH.get(card.card_id,card.name_en),
                         english=card.name_en,kind=card.kind,image=picture))
    return rows
