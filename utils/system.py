import platform


def get_os_name():
    return platform.system()


if __name__ == '__main__':
    print(get_os_name())
