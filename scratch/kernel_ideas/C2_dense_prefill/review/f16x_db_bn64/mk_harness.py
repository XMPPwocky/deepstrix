# Copy the engineer's harness.cpp into review/ with an extended correctness tail list (b values the
# engineer did not test: 7, 63, 64, 65, 191, 511, 513, 1000, 1023). Nothing else changes.
import pathlib
src = pathlib.Path("../../harness.cpp").read_text()
old = "for (unsigned t : {1u, 3u, 17u, 100u, 129u, 255u, 256u, 500u}) if (t < b) bs.push_back(t);"
new = ("for (unsigned t : {1u, 3u, 7u, 17u, 63u, 64u, 65u, 100u, 129u, 191u, 255u, 256u, 500u, "
       "511u, 512u, 513u, 1000u, 1023u}) if (t < b) bs.push_back(t);")
assert src.count(old) == 1
src = src.replace(old, new)
pathlib.Path("harness_review.cpp").write_text(src)
print("ok")
