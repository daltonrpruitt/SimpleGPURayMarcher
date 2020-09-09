from pynput import keyboard


def press_callback(key):
    #print("Key pressed: {0}".format(key))
    if key == keyboard.Key.esc:
        return False


def rel_callback(key):
    pass

with keyboard.Listener(on_press=press_callback, rel_press=rel_callback) as listener:
    listener.join()