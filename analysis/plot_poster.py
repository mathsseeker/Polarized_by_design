from PIL import Image

# Load both screenshots
img1 = Image.open("not_polarized.png")  # Replace with your filename
img2 = Image.open("polarized.png")


# Resize to the same height, if needed
h = min(img1.height, img2.height)
img1 = img1.resize((int(img1.width * h / img1.height), h), Image.LANCZOS)
img2 = img2.resize((int(img2.width * h / img2.height), h), Image.LANCZOS)

# Combine side by side
combined = Image.new("RGB", (img1.width + img2.width, h), (244,236,216))  # use your poster bg color if desired
combined.paste(img1, (0, 0))
combined.paste(img2, (img1.width, 0))

# Save to current folder—no subdirectory!
combined.save("fig_network_panels.png")