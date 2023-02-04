res = {
  "OpeningToHall": [
    1,
    1,
    1
  ],
  "OpeningToRoom": [
    2,
    2,
    2
  ],
  "Closet": [
    3,
    3,
    3
  ],
  "Bathroom": [
    4,
    4,
    4
  ],
  "Hall": [
    5,
    5,
    5
  ],
  "Balcony": [
    6,
    6,
    6
  ],
  "Window": [
    7,
    7,
    7
  ],
  "Background": [
    8,
    8,
    8
  ],
  "Room": [
    9,
    9,
    9
  ],
  "Wall": [
    10,
    10,
    10
  ],
  "Opening": [
    11,
    11,
    11
  ],
  "Door": [
    12,
    12,
    12
  ],
  "Utilities": [
    13,
    13,
    13
  ]
}

new_d = dict()
for key, value in res.items():
    new_d[key.lower()] = value[0]
print(new_d)