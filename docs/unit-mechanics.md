# 卡牌特殊机制覆盖表（3.12.10）

本表由本地 `data/battle_knowledge.json` 和执行代码生成，共 132 张卡牌。已执行表示进入了模拟计算，并不表示已经完成当前游戏版本的数值、物理时序或胜率验收。

## 本次实现

- 超骑：按照卡牌登场投射物计算一次范围伤害，按单位属性计算跃击距离、蓄力、飞行、落地伤害与击退；已存在的观测单位不会重新触发登场伤害。
- 幻影刺客：按数据库冲刺距离和伤害计算，飞行阶段应用伤害免疫；不把进化/英雄主动技能混入普通卡。
- 电法/冰法：读取登场区域效果的等级伤害、目标类型、眩晕或减速；电法双电击可分配给两个目标，只有一个目标时两次命中该目标。
- 电巨人：只有带攻击来源、且攻击者位于反击范围内的命中触发反击；普通法术不伪造攻击者。
- 电龙：按跳转半径与最大目标数连接不同目标，保留逐段传播时延估计。
- 迫击炮等：最小射程参与目标选择。冲锋、护盾、死亡伤害、召唤和圣水生产沿用原执行器，并补充缺失定义审计。

## 数值与模拟限制

优先使用明确等级表。跃击/反击等只有基础值的字段，根据原始单位等级生命值比例缩放取整，不用混合来源的当前生命值覆盖原始缩放；该结果属于估计。数据库来源包含历史机制与公开详情数值，未核验当前平衡。跃击飞行路径、移动目标落点、冲刺免疫完整边界及连锁传播时延仍是空间近似。未导入的死亡炸弹、附属乘员、回旋/穿透弹道、隐身、散弹及进化/英雄主动能力等在表中保留待支持标记。

落点筛选加入登场与跃击的粗评分，CPU/GPU筛选均应用相同附加评分；实际动作仍由完整时间推进模拟比较。模拟版本为 `spatial_v6_grid_unit_mechanics`，无需修改模型权重或保存的策略键。

官方说明区分超骑登场和跃击，并记录过独立的登场伤害调整：[Supercell 2022 年平衡说明](https://supercell.com/en/games/clashroyale/blog/release-notes/balance-changes-april-2022/)。这里只用来核对机制区别，不把历史调整当成当前数值。

## 全部卡牌

“—”表示该列没有特殊机制记录；普通攻防不在已执行列逐项列出。待支持列保留数据库/执行器字段名，便于定位实现。

| 卡牌 | 已执行的特殊机制 | 待支持或待核验 |
|---|---|---|
| knight | — | evolution_or_hero_variant |
| archers | — | evolution_or_hero_variant |
| goblins | — | evolution_or_hero_variant |
| giant | — | evolution_or_hero_variant |
| pekka | — | evolution_or_hero_variant |
| minions | — | — |
| balloon | 死亡伤害 | death_spawn_missing_definition |
| witch | 周期召唤 | evolution_or_hero_variant |
| barbarians | — | evolution_or_hero_variant |
| golem | 死亡伤害、死亡衍生单位 | — |
| skeletons | — | evolution_or_hero_variant |
| valkyrie | — | evolution_or_hero_variant |
| skeleton_army | — | evolution_or_hero_variant |
| bomber | — | evolution_or_hero_variant |
| musketeer | — | evolution_or_hero_variant |
| baby_dragon | — | evolution_or_hero_variant |
| prince | 冲锋 | jump_speed |
| wizard | — | evolution_or_hero_variant |
| mini_pekka | — | evolution_or_hero_variant |
| spear_goblins | — | — |
| giant_skeleton | 死亡伤害 | death_spawn_missing_definition |
| hog_rider | — | jump_speed |
| minion_horde | — | — |
| ice_wizard | 登场伤害/控制 | — |
| royal_giant | — | evolution_or_hero_variant |
| guards | 护盾 | — |
| princess | — | multiple_projectiles |
| dark_prince | 冲锋、护盾 | jump_speed |
| three_musketeers | — | — |
| lava_hound | 死亡衍生单位 | — |
| ice_spirit | — | evolution_or_hero_variant、kamikaze |
| fire_spirit | — | kamikaze |
| miner | — | — |
| sparky | — | — |
| bowler | — | projectile:projectile_range |
| lumberjack | — | death_spawn_missing_definition、evolution_or_hero_variant |
| battle_ram | 冲锋、死亡衍生单位 | evolution_or_hero_variant、kamikaze |
| inferno_dragon | — | evolution_or_hero_variant |
| ice_golem | 死亡伤害 | evolution_or_hero_variant |
| mega_minion | — | evolution_or_hero_variant |
| dart_goblin | — | evolution_or_hero_variant |
| goblin_gang | — | — |
| electro_wizard | 多目标攻击、登场伤害/控制、眩晕 | buff_on_damage |
| elite_barbarians | — | — |
| hunter | — | evolution_or_hero_variant、multiple_projectiles、projectile:projectile_range |
| executioner | — | evolution_or_hero_variant、projectile:pingpong_visual_time、projectile:projectile_range |
| bandit | 跃击/冲刺 | — |
| royal_recruits | 护盾 | evolution_or_hero_variant |
| night_witch | 死亡衍生单位、周期召唤 | — |
| bats | — | evolution_or_hero_variant |
| royal_ghost | — | attached_character、evolution_or_hero_variant、invisibility |
| ram_rider | 冲锋、周期召唤 | jump_speed |
| zappies | 眩晕 | buff_on_damage |
| rascals | — | — |
| cannon_cart | 死亡衍生单位 | — |
| mega_knight | 登场范围伤害、跃击/冲刺 | evolution_or_hero_variant |
| skeleton_barrel | 死亡伤害 | death_spawn_missing_definition、evolution_or_hero_variant、kamikaze |
| flying_machine | — | — |
| wall_breakers | — | evolution_or_hero_variant、kamikaze、projectile:projectile_range |
| royal_hogs | — | evolution_or_hero_variant、jump_speed |
| goblin_giant | 周期召唤 | evolution_or_hero_variant |
| fisherman | — | projectile_special |
| magic_archer | — | evolution_or_hero_variant、projectile:projectile_range |
| electro_dragon | 连锁攻击 | evolution_or_hero_variant |
| firecracker | — | evolution_or_hero_variant、projectile:spawn_projectile |
| mighty_miner | — | — |
| super_witch | 周期召唤 | spawn_character2 |
| elixir_golem | 死亡衍生单位 | — |
| battle_healer | 登场伤害/控制 | area_effect_on_hit |
| skeleton_king | — | — |
| super_lava_hound | 死亡衍生单位 | death_spawn_projectile |
| super_magic_archer | — | projectile:projectile_range |
| archer_queen | — | — |
| santa_hog_rider | 周期召唤 | jump_speed |
| golden_knight | — | dash_chain_or_ability、jump_speed |
| super_ice_golem | 死亡伤害 | — |
| monk | — | — |
| super_archers | — | projectile:projectile_range |
| skeleton_dragons | — | — |
| terry | — | dash_chain_or_ability、jump_speed |
| super_mini_pekka | 周期召唤 | buff_on_damage |
| mother_witch | — | buff_on_damage |
| electro_spirit | 连锁攻击 | kamikaze |
| electro_giant | 受击反击 | — |
| raging_prince | 冲锋 | jump_speed |
| phoenix | 死亡伤害 | death_spawn_projectile |
| cannon | — | evolution_or_hero_variant |
| goblin_hut | 死亡衍生单位、周期召唤 | — |
| mortar | 登场范围伤害、最小射程 | evolution_or_hero_variant |
| inferno_tower | — | — |
| bomb_tower | 死亡伤害、登场范围伤害 | attached_character、death_spawn_missing_definition |
| barbarian_hut | 死亡衍生单位、周期召唤 | — |
| tesla | — | evolution_or_hero_variant、hides_when_not_attacking、invisibility |
| elixir_collector | 圣水生产 | — |
| x_bow | — | — |
| tombstone | 死亡衍生单位、周期召唤 | — |
| furnace | 死亡衍生单位、周期召唤 | evolution_or_hero_variant |
| goblin_cage | 死亡衍生单位 | evolution_or_hero_variant |
| goblin_drill | — | evolution_or_hero_variant |
| party_hut | 死亡衍生单位 | attached_character、death_spawn_projectile |
| fireball | — | — |
| arrows | spell_waves | — |
| rage | spell_rage | — |
| rocket | — | — |
| goblin_barrel | spell_spawn | evolution_or_hero_variant、spawn_character |
| freeze | — | area_effect_object |
| mirror | — | unmapped_combat_mechanics |
| lightning | spell_highest_hp | area_effect_object、target_buff |
| zap | — | area_effect_object、evolution_or_hero_variant |
| poison | spell_periodic | area_effect_object |
| graveyard | spell_serial_spawn | area_effect_object、spawn_character |
| the_log | spell_line | projectile_range |
| tornado | spell_pull | area_effect_object |
| clone | spell_clone | area_effect_object |
| earthquake | spell_earthquake | area_effect_object |
| barbarian_barrel | spell_line_spawn | evolution_or_hero_variant、projectile_range、spawn_character |
| heal_spirit | — | kamikaze |
| giant_snowball | — | evolution_or_hero_variant、target_buff |
| royal_delivery | spell_area_spawn | spawn_character |
| party_rocket | — | unmapped_combat_mechanics |
| little_prince | — | unmapped_combat_mechanics |
| goblin_demolisher | — | unmapped_combat_mechanics |
| goblin_machine | — | unmapped_combat_mechanics |
| suspicious_bush | — | unmapped_combat_mechanics |
| goblinstein | — | unmapped_combat_mechanics |
| rune_giant | — | unmapped_combat_mechanics |
| berserker | — | detail_base_attack_unvalidated_balance、unmapped_combat_mechanics |
| boss_bandit | — | unmapped_combat_mechanics |
| void | — | unmapped_combat_mechanics |
| goblin_curse | — | unmapped_combat_mechanics |
| spirit_empress | — | unmapped_combat_mechanics |
| vines | — | unmapped_combat_mechanics |
