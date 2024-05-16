import os
# Un module pour interagir avec le GUIX profile est ses variables
# d'environnement.
PROFILE = str(os.getenv("GUIX_LOAD_PROFILE"))
iddfile = "./V9-5-0-Energy+.idd"
def weatherfile(name):
    return PROFILE + "/share/weatherfiles/" + name
