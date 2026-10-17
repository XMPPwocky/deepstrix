Verbatim copies of `encoding/tests/test_input_{1..5}.json` and
`encoding/tests/test_output_{1..5}.txt` from the DeepSeek-V4.1-Flash-1017
Hugging Face checkpoint (local copy `/weights2/dsv4.1f-1017/encoding/tests/`,
fetched 2026-10-17; byte-identical to the original V4.1-Flash release's
copies). MIT License, Copyright (c) 2023 DeepSeek (the checkpoint's
`LICENSE`).

The outputs are what the checkpoint's `encoding/encoding.py` renders for each
input (`encode_case(case, thinking_mode="chat")`). Used by
`tests/oracle_prompt.rs::official_vectors_*`.

sha256:
6a5a86694e47169c8f0c9243a7e24cc5db05447be0d3d3a9891034e913c518de  test_input_1.json
fd985cda88e8640b8e2404d8e659f7006cad204eb41f73c9e6662b1f218c01aa  test_input_2.json
4207613f15c79b95ad92074bdb8b2b81c5ce9cfdf15b7a67f58ded6bb3ec4fa5  test_input_3.json
4c3890a7bc042ec6396d98d024a39a73cdcffbe1d0253e852da0c35e65296f3e  test_input_4.json
88289b0aac8d650aa55c4ffeb392818fff3a0ac83c2293ddfd90546804d5e5c4  test_input_5.json
91f79aa955717b414a82aabfb38bcd83ab5a998bd47db89d2a26338e8d3de550  test_output_1.txt
9fa162587b73ab9b315be5c88d433f9e0e681b4b36926952f289099ed9bc1dab  test_output_2.txt
cf2b801d0738c88abb691bf4a096000c0e0737faab8a54a4ba5243c5513a3711  test_output_3.txt
94433b68370dcc69fa61efc5011b5d0e28685e4716fee1ee77fbbfc0c9935b54  test_output_4.txt
4d673091746553d1cb2e40a3fc01dc09a7c53c40627096a3e49c97a269a81f64  test_output_5.txt
