CLUSTERS: dict[str, list[str]] = {
  "object_actions": {
    r"\badjust(s|ed|ing)?\s+(?:the\s+)?(?:tie|jacket|coat|collar|glasses|hood|robe|doublet|quill)\b",
    r"\bwipe(s|d|ing)?\s+(?:the\s+)?(?:quill|blade|face|hands?)\b",
    r"\bburn(s|ed|ing)?\s+(?:it|them|the\s+\w+)\b",
    r"\bdon(s|ned|ning)?\s+(?:the\s+)?(?:tie|jacket|coat|hood|robe|doublet|armor|armour|corporate\s+suit)\b",
    r"\bremove(s|d|ing)?\s+(?:the\s+)?(?:tie|jacket|coat|hood|robe|doublet|armor|armour|corporate\s+suit)\b",
    r"\bstraighten(s|ed|ing)?\s+(?:the\s+)?(?:tie|jacket|coat|shirt)\b",
  },
  "objects": {
    r"\bquill\b",
    r"\bdoorframe\b",
    r"\bincinerator\b",
  }
}