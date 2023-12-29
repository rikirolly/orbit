from skyfield.api import load
from astroquery.jplhorizons import Horizons
from astropy import units as u
import re


def extract_data(text, id):
    # Adjusted regex to extract the planet name
    planet_name_pattern = r"\b([A-Za-z]+)\s+{}\b".format(id)
    planet_name_match = re.search(planet_name_pattern, text)
    planet_name = planet_name_match.group(1) if planet_name_match else "Unknown"

    # Regex to extract the mass
    mass_pattern = r"Mass x10\^24 \(kg\)= ([\d.+-]+)"
    mass_match = re.search(mass_pattern, text)
    mass = mass_match.group(1) if mass_match else "Unknown"

    return {
        "planet_name": planet_name,
        "mass_x10_24_kg": mass
    }

for object_id in range(199, 999, 100):

    obj_id = Horizons(id=object_id, location="0") # Location 0 is the solar system barycenter
    response = obj_id.ephemerides_async()
    print(extract_data(response.text, object_id))


planets = load('de421.bsp')
earth, mars = planets['earth'], planets['mars']
sun = planets['sun']
ts = load.timescale()
t = ts.now()
position = sun.at(t).observe(earth)
ra, dec, distance = position.radec()