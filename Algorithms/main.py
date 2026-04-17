from memory import add_memory, retrieve

add_memory("i like ice creams")
add_memory("my fav driver is oscar piastri")

res = retrieve("prefrence", k = 2 )

print(res["prefrence"])
