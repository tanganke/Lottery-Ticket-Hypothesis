from _common import *

log = logging.getLogger(__name__)

# imports


@hydra.main(str(CONFIG_DIR), "config", None)
def main(cfg: DictConfig):
    print(cfg)

if __name__ == '__main__':
    main()
