from __future__ import annotations

import hashlib
import json
import tempfile
import tkinter as tk
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from crbot.card_library import CardLibrary, encode_json
from crbot.cards import CardCatalog


class LibraryFixture(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root_path = Path(self.directory.name)
        (self.root_path / "data").mkdir()
        self.catalog_path = self.root_path / "data/cards.json"
        ids = ["knight", "archers", "skeletons", "skeleton_army", "graveyard", "golem",
               "goblin_gang", "goblins", "spear_goblins", "three_musketeers", "musketeer"]
        self.catalog_path.write_bytes(encode_json({"schema_version": 1, "source": "fixture",
            "unrelated": {"preserve": True}, "cards": [dict(id=cid, name_en=cid.title(),
                name_zh="骑士" if cid == "knight" else "", kind="troop", elixir=3,
                detail_stats={"hitSpeed": 1.2}, custom_field="preserve") for cid in ids]}))
        (self.root_path / "data/deckbuilder_cards.json").write_bytes(encode_json({"cards": [
            {"id": "knight", "source_id": "knight", "battle_stats": {"hitSpeed": 1.2}}]}))
        self.slices = self.root_path / "training/external/katacr-slices"
        self.slices.mkdir(parents=True)
        records = []
        for unit in ("knight", "knight-evolution", "archer", "skeleton", "skeleton-evolution",
                     "golem", "golemite", "goblin", "spear-goblin", "musketeer", "bar", "queen-tower"):
            path = self.slices / "sprites" / unit / f"{unit}_0_1.png"
            path.parent.mkdir(parents=True)
            Image.new("RGBA", (24, 32), "red").save(path)
            records.append(dict(unit=unit, path=path.relative_to(self.slices).as_posix(),
                affiliation=0, size=[24, 32], sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        (self.slices / "manifest.json").write_bytes(encode_json(dict(source="github-fixture", revision="pinned",
            records=records, auxiliary=[], classes=[r["unit"] for r in records])))
        (self.slices / "LICENSE").write_text("fixture MIT")
        self.icon = self.root_path / "icon.png"
        Image.new("RGBA", (32, 40), "blue").save(self.icon)
        self.lib = CardLibrary(self.root_path)


class CardLibraryTests(LibraryFixture):
    def units(self, card):
        return {r["unit"] for r in self.lib.by_card[card]}

    def test_exact_evolution_shared_and_companion_relationships(self):
        self.assertEqual(self.units("knight"), {"knight", "knight-evolution"})
        self.assertEqual(self.units("golem"), {"golem", "golemite"})
        self.assertEqual(self.units("skeleton_army"), {"skeleton"})
        self.assertEqual(self.units("skeletons"), {"skeleton", "skeleton-evolution"})
        self.assertEqual(self.units("goblin_gang"), {"goblin", "spear-goblin"})
        self.assertEqual(self.units("three_musketeers"), {"musketeer"})
        self.assertEqual({r["unit"] for r in self.lib.unmapped}, {"bar", "queen-tower"})
        self.assertEqual(self.lib.search("骑士")[0].card_id, "knight")
        self.assertIn("knight", [c.card_id for c in self.lib.search("knight-evolution")])

    def test_local_batch_validated_deduplicated_and_shared_links_preserved(self):
        before = (self.slices / "manifest.json").read_bytes()
        original = next(r for r in self.lib.records if r["unit"] == "knight")
        self.assertEqual(self.lib.add_slices([original["file"]], "knight", 0, ["knight"]), 0)
        count = self.lib.add_slices([self.icon, self.icon], "goblin", 1, ["knight"])
        self.assertEqual(count, 1)
        self.assertEqual(set(self.lib.links["goblin"]), {"goblins", "goblin_gang", "knight"})
        self.assertEqual(self.lib.add_slices([self.icon], "goblin", 1, ["knight"]), 0)
        self.assertEqual((self.slices / "manifest.json").read_bytes(), before)
        bad = self.root_path / "bad.png"
        Image.new("RGB", (4, 4), "red").save(bad)
        with self.assertRaisesRegex(ValueError, "透明"):
            self.lib.add_slices([self.icon, bad], "new-class", 0, ["knight"])
        self.assertNotIn("new-class", self.lib.classes)
        with self.assertRaises(ValueError):
            self.lib.add_slices([self.icon], "../escape", 0, ["knight"])

    def test_add_card_preserves_catalog_and_recovers_backup(self):
        before = self.catalog_path.read_bytes()
        self.lib.add_card(card_id="new_card", name_zh="新卡", name_en="New Card", elixir=4,
                          kind="troop", rarity="rare", icon=self.icon)
        after = json.loads(self.catalog_path.read_bytes())
        self.assertEqual(after["unrelated"], {"preserve": True})
        self.assertEqual(after["cards"][:-1], json.loads(before)["cards"])
        loaded = CardCatalog.load(self.catalog_path)
        self.assertTrue(loaded.local_icon_path(loaded.get("new_card")).is_file())
        self.assertEqual(next((self.lib.local_path / "backups").glob("*.json")).read_bytes(), before)
        with self.assertRaisesRegex(ValueError, "已存在"):
            self.lib.add_card(card_id="new_card", name_zh="重复", name_en="", elixir=1,
                              kind="troop", rarity="common", icon=self.icon)

    def test_explicit_links_survive_reload_and_missing_import_is_browsable(self):
        self.lib.set_links("bar", ["knight", "golem"])
        reopened = CardLibrary(self.root_path)
        self.assertEqual(reopened.links["bar"], ["golem", "knight"])
        reopened.set_links("bar", [])
        self.assertIn("bar", {r["unit"] for r in reopened.unmapped})
        missing = CardLibrary(self.root_path, slices_path=self.root_path / "absent")
        self.assertEqual(len(missing.catalog.cards), 11)
        self.assertEqual(missing.classes, [])

    def test_export_is_self_contained_keeps_classes_and_detects_corruption(self):
        self.lib.add_slices([self.icon], "new-unit", 1, ["knight"])
        output = self.root_path / "export"
        result = self.lib.export_snapshot(output)
        self.assertFalse(result["promoted"])
        self.assertFalse(result["battle_acceptance"])
        self.assertTrue(result["library_customized"])
        self.assertEqual(result["card_class_links"]["new-unit"], ["knight"])
        for row in result["records"]:
            self.assertEqual(hashlib.sha256((output / row["path"]).read_bytes()).hexdigest(), row["sha256"])
        self.assertEqual((output / "LICENSE").read_text(), "fixture MIT")
        with self.assertRaises(ValueError):
            self.lib.export_snapshot(output)
        self.lib.records[0]["file"].write_bytes(b"corrupted")
        with self.assertRaisesRegex(ValueError, "校验失败"):
            self.lib.export_snapshot(self.root_path / "bad-export")
        self.assertFalse((self.root_path / "bad-export").exists())

    def test_export_consumed_by_existing_synthetic_pipeline_with_disjoint_splits(self):
        from crbot.slice_dataset import export_synthetic
        paths = []
        for i in range(20):
            path = self.root_path / f"sample-{i}.png"
            Image.new("RGBA", (20, 30), (i*10, 40, 80, 255)).save(path)
            paths.append(path)
        self.lib.add_slices(paths, "knight", 1, ["knight"])
        snapshot = self.root_path / "snapshot"
        self.lib.export_snapshot(snapshot)
        exported = export_synthetic(snapshot, self.root_path / "synthetic")
        self.assertGreater(exported["counts"]["train"], 0)
        self.assertGreater(exported["counts"]["val"], 0)
        self.assertFalse(set(exported["source_hashes"]["train"]) & set(exported["source_hashes"]["val"]))

    def test_manifest_cannot_read_outside_the_library(self):
        manifest = json.loads((self.slices / "manifest.json").read_bytes())
        manifest["records"][0]["path"] = "../../../icon.png"
        (self.slices / "manifest.json").write_bytes(encode_json(manifest))
        with self.assertRaisesRegex(ValueError, "超出"):
            self.lib.reload()


class CardLibraryUITests(LibraryFixture):
    def test_add_card_then_add_slice_through_dialog_actions(self):
        from crbot.ui_widgets import HoverButton
        def descend(widget):
            for child in widget.winfo_children():
                yield child
                yield from descend(child)
        def dialog():
            return next(c for c in self.ui.window.winfo_children() if isinstance(c, tk.Toplevel))
        with patch("crbot.card_library_ui.filedialog.askopenfilename", return_value=str(self.icon)), \
             patch("crbot.card_library_ui.messagebox.showerror") as error:
            self.ui.add_card_dialog()
            window = dialog()
            entries = [w for w in descend(window) if isinstance(w, tk.Entry)]
            for widget, value in zip(entries, ["custom_card", "自定义卡牌", "Custom Card", "4"]):
                widget.delete(0, "end")
                widget.insert(0, value)
            actions = {w.cget("text"): w for w in descend(window) if isinstance(w, HoverButton)}
            actions["选择卡牌图标"].invoke()
            actions["保存卡牌"].invoke()
            self.assertFalse(error.called)
            self.assertEqual(self.ui.card_id, "custom_card")
        with patch("crbot.card_library_ui.filedialog.askopenfilenames", return_value=[str(self.icon)]), \
             patch("crbot.card_library_ui.messagebox.showinfo"), \
             patch("crbot.card_library_ui.messagebox.showerror") as error:
            self.ui.add_slices_dialog()
            window = dialog()
            actions = {w.cget("text"): w for w in descend(window) if isinstance(w, HoverButton)}
            actions["选择切片图片"].invoke()
            actions["添加到当前卡牌"].invoke()
            self.assertFalse(error.called)
            self.assertEqual(len(self.ui.library.by_card["custom_card"]), 1)
        self.assertEqual(CardLibrary(self.root_path).catalog.get("custom_card").name_zh, "自定义卡牌")
        self.assertFalse(self.errors)

    def test_main_entry_reuses_window_and_opens_configured_catalog(self):
        from crbot.gui import RoyalTrainerApp
        config = self.root_path / "config.json"
        config.write_text(json.dumps({"game": {"allowed_mode": "offline_ai_only"},
                                     "dataset": {"card_catalog": "data/cards.json"}}))
        with patch.object(self.root, "after"):
            app = RoyalTrainerApp(self.root, config_path=config, start_services=False)
        app.card_library_button.invoke()
        first = app.card_library_window
        self.assertEqual(first.library.catalog_path, self.catalog_path.resolve())
        app.card_library_button.invoke()
        self.assertIs(app.card_library_window, first)
        first.window.destroy()
        app.card_library_button.invoke()
        self.assertIsNot(app.card_library_window, first)
        self.assertFalse(self.errors)

    def setUp(self):
        super().setUp()
        self.root = tk.Tk()
        self.root.withdraw()
        self.errors = []
        self.root.report_callback_exception = lambda *args: self.errors.append(str(args[1]))
        self.addCleanup(self.root.destroy)
        from crbot.card_library_ui import CardLibraryWindow
        with patch.dict("os.environ", {"CRBOT_REDUCED_MOTION": "1"}):
            self.ui = CardLibraryWindow(self.root, self.lib)
        self.root.update()

    def test_click_tile_filter_then_return_to_cards(self):
        tile = self.ui.tiles.winfo_children()[0]
        self.assertIsNotNone(tile.photo)
        tile.invoke()
        self.root.update()
        self.assertEqual(self.ui.card_id, "knight")
        self.ui.group.set("knight-evolution")
        self.ui.filter_changed()
        self.assertEqual(len(self.ui.rows()), 1)
        self.ui.show_cards()
        self.root.update()
        self.assertEqual(len(self.ui.rows()), 11)
        self.assertFalse(self.errors)

    def test_unmapped_and_dialogs_open_without_callback_errors(self):
        self.ui.show_unmapped()
        self.root.update()
        self.assertEqual(len(self.ui.rows()), 2)
        self.ui.show_card("knight")
        for command in (self.ui.add_card_dialog, self.ui.add_slices_dialog,
                        self.ui.links_dialog, self.ui.metadata_dialog):
            command()
            self.root.update()
            for child in self.ui.window.winfo_children():
                if isinstance(child, tk.Toplevel):
                    child.destroy()
        self.assertFalse(self.errors)

    def test_pagination_loads_only_visible_images_and_minimum_geometry_fits(self):
        self.ui.PAGE_SIZE = 4
        self.ui.render()
        self.assertEqual(len(self.ui.photos), 4)
        self.ui.change_page(1)
        self.assertEqual(self.ui.page, 1)
        self.assertEqual(len(self.ui.photos), 4)
        self.ui.window.geometry("1000x700+0+0")
        self.root.update()
        for widget in (self.ui.next_button, self.ui.prev_button, self.ui.search_entry):
            self.assertTrue(widget.winfo_ismapped())
            self.assertGreater(widget.winfo_height(), 10)
            self.assertLessEqual(widget.winfo_rootx() - self.ui.window.winfo_rootx()+widget.winfo_width(), 1000)
            self.assertLessEqual(widget.winfo_rooty() - self.ui.window.winfo_rooty()+widget.winfo_height(), 700)
        self.assertFalse(self.errors)


if __name__ == "__main__":
    unittest.main()
