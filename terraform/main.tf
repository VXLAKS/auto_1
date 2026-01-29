terraform {
  required_providers {
    yandex = { source = "yandex-cloud/yandex" }
  }
}

provider "yandex" {
  token     = "fake_token_for_assignment"
  cloud_id  = "fake_cloud_id"
  folder_id = "fake_folder_id"
  zone      = "ru-central1-a"
}

# Сеть
resource "yandex_vpc_network" "internal" {
  name = "internal-network"
}

# Подсеть
resource "yandex_vpc_subnet" "internal-a" {
  network_id     = yandex_vpc_network.internal.id
  v4_cidr_blocks = ["10.1.0.0/24"]
  zone           = "ru-central1-a"
}

# Кластер Kubernetes
resource "yandex_kubernetes_cluster" "scoring_cluster" {
  name       = "scoring-cluster"
  network_id = yandex_vpc_network.internal.id

  master {
    zonal {
      zone      = yandex_vpc_subnet.internal-a.zone
      subnet_id = yandex_vpc_subnet.internal-a.id
    }
  }
}