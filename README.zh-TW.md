# aria-memory（繁體中文說明）

> 本專案屬於研究《常駐型個人 AI 語音助理：系統架構與新型記憶系統研究（附：大型語言模型的時間感知研究）》，這裡是其中的「新型記憶系統」。總入口是 [always-on-assistant](https://github.com/Alex20090906/always-on-assistant)。
>
> English: [README.md](README.md)

**給常駐型語音助理用的記憶：使用者說過的話寫成只增不改的紀錄；網路權重裡的稀疏聯想記憶（海馬迴式索引）從字和讀音找出相關紀錄；讀的人只在紀錄明確講的是同一件事時才回答。另外附一個閘門，讓助理可以自己想事情，又不會陷入自我循環（植物人狀態）。**

這是研究程式碼，純 Python 與 numpy，用 CPU 就能跑。負責聆聽和讀紀錄的 LLM 由你提供：任何 `async ask(messages, max_tokens) -> str` 的函式都可以。

## 快速試玩（不需要 GPU、不用下載模型，約 10 秒）

```bash
git clone https://github.com/Alex20090906/aria-memory && cd aria-memory
python -m venv .venv && . .venv/bin/activate      # Windows：.venv\Scripts\activate
pip install -e ".[test]"
python examples/quickstart.py
pytest -q
```

`examples/quickstart.py` 會：
1. 用幾筆口語紀錄建一個索引並查詢；
2. 用嚴格評分重算定案系統（E49）存下的答案：講過的 51 題答對 30、答錯 6，沒講過的 21 題守住 20 題；
3. 在兩份 21 天測試腳本上跑索引，看正確紀錄有沒有進前 5 筆；
4. 算出只聽喚醒詞時，還有多少內容聽得到。

它用的是 `CharVocab`，不需要任何模型的詞表。`pytest -q`：沒有模型詞表時 48 個通過、4 個略過；有詞表時（見 [docs/VOCAB.md](docs/VOCAB.md)）52 個全部通過，並且和研究數字完全一致。

## 結果

所有輸入都是合成的：LLM 寫的腳本、Windows 文字轉語音的聲音、程式產生的室內雜音，再用真的語音辨識聽寫。**沒有用到任何真人的聲音或生活。** 每個數字的出處在 [docs/research/](docs/research/)。

| 結果 | 數值 | 資料 | 重跑方式或紀錄位置 |
|---|---|---|---|
| 講過的問題答對／答錯（定案系統 E49，模擬 21 天，嚴格評分） | 30／6（共 51） | 合成語音 | `python examples/quickstart.py` |
| 沒講過的問題回答「不知道」 | 20／21 | 合成語音 | 同上 |
| 正確紀錄進索引前 5 筆（模型詞表） | 35/38、34/38 | 合成語音 | `tests/test_index.py::TestRealVocab`（需要詞表）；紀錄 `experiments/p8/eval_h41_ortho.txt` |
| 同上，`CharVocab`（不用模型） | 34/38、34/38 | 合成語音 | `python examples/quickstart.py` |
| 加 1 萬筆同名困難干擾：網路索引 vs 文字向量模型 | 18 vs 15、29 vs 14 | 合成語音 | 紀錄 `experiments/p8/eval_index_scale_dense.txt` |
| 長期儲存：2,400 個事件中的 60 個事實，答對／誠實忘記／有把握答錯 | 56／4／0（最舊四分之一 15/15） | 合成文字 | 紀錄 `experiments/cloud/results_modal/sleep_never_n2340_u15_never_echo_2400.json`；重新評分 `experiments/cloud/rescore_output.txt` |
| 同樣負載，內容放在快權重＋睡眠鞏固 | 15／18／27 | 合成文字 | 紀錄 `experiments/cloud/results_gputw/load2400__sleep_sleep_n2340_load2400.json` |
| 每次讀取回答的時間，中位數／90 百分位 | 1.54／1.90 秒 | 思考模型：RTX 3080 上的 Qwen3.8-Flash-Next | 紀錄 `experiments/alwayson/latency_table.txt` |
| 21 天中的植物人小時：受閘門約束 vs 無煞車 | 504 小時中 2 vs 72 小時中 72 | 模擬日 | 紀錄 `experiments/alwayson/results/summary.md` |
| 強迫連續生成：加重複懲罰時答對的提問 | 2/9（不流暢度 0.40 → 2.80） | 合成文字 | 紀錄 `experiments/habituation/results_a/`、報告 `experiments/阶段报告_01.md` |
| 只聽喚醒詞時還聽得到的講過問題；加 8 秒跟隨視窗 | 24/51；44/51 | 合成語音 | `python examples/quickstart.py` |

標示「紀錄」的列需要思考模型（78 GB，要 GPU），這裡沒有重跑；原始輸出都在 repo 裡，評分程式用 CPU 就能跑。

## 運作方式

| 模組（`src/aria_memory/`） | 工作 |
|---|---|
| `listen.py`、`rules.py` | 使用者的一句話 → 紀錄（東西｜方面｜內容＋原話）；暫存 2 句讓改口能取代；相對日期由規則換算；清掉被寫成事實的問句 |
| `index.py`、`vocab.py` | 索引：65,536 個單元的稀疏鍵（意義單元來自模型第 0 層詞向量，或用 `CharVocab`；加上拼音讀音單元）、512 維紀錄編碼、差值規則寫入、存檔、真的刪除 |
| `read.py`、`records.py` | 紀錄怎麼給讀的人看；先找再讀，名稱讀音完全相同時再讀一次 |
| `sleep.py` | 睡前重聽當天的話補漏；嚴格標記被取代的舊紀錄 |
| `gate.py`、`reflex_gru.py` | 常駐閘門（能量池、四個熔斷、閒置工作、主動開口上限）；研究用的反射層模型（numpy 版） |
| `eval.py`、`replay.py` | 21 天腳本、嚴格評分、喚醒詞涵蓋率、快取重播、驗收報告 |

提示詞是繁體中文（`prompts.py`），研究目標是台灣華語家庭。提示詞裡助理的名字與預設喚醒詞是 `NAME = "Aria"`。接自己的 LLM 的完整範例：`python examples/with_your_llm.py`。

## 和 Letta、mem0、Graphiti 的差異

那些是給聊天代理用的完整記憶產品；這裡是給常駐語音助理的研究程式。這裡的紀錄永遠不改寫；找紀錄靠網路權重裡的聯想記憶（字與讀音，不需要另外的向量服務）；沒有紀錄明確講到問題時，讀的人必須說不知道；評測保留語音辨識錯字，並計算「沒講過的事有沒有誠實說不知道」；還有一個閘門限制助理自己想事情的量。比較表與注意事項見 [docs/research/08-limitations.md](docs/research/08-limitations.md)。

## 公開基準

LongMemEval、LoCoMo 等公開長期記憶基準：**未跑**。任務的差異與怎麼跑，見 [docs/research/08-limitations.md](docs/research/08-limitations.md)。

## 限制

- 只有合成資料：兩份 21 天華語腳本、三個 TTS 聲音。聲紋相關數字目前沒有意義。
- 大多只跑一個種子；51 題裡差 2 題以內算雜訊。
- 端到端只量過一個思考模型；常駐迴圈在模擬器裡跑，沒有接真的裝置。
- 儲存最多測到 2,400 個事件，幾年的使用沒測過。
- 快權重引擎的實驗需要打過補丁的 Strata 引擎（`third_party/strata/`）和 GPU。

## 授權

Apache-2.0（[LICENSE](LICENSE)），`third_party/strata/` 例外（MIT，沿用 Strata 的授權）。不含模型權重與音檔，見 [NOTICE](NOTICE) 與 [docs/REVIEW.md](docs/REVIEW.md)。研究原始紀錄（`experiments/`）以繁體中文為主。
