#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // for std::vector support
#include <fstream>
#include "seal/seal.h"
#include "seal/modulus.h"
#include "seal/serialization.h"

#pragma message("SEAL Version = " SEAL_VERSION)

namespace py = pybind11;
using namespace seal;
using namespace seal; 
using namespace std;

PYBIND11_MODULE(seal_wrapper, m) {
    //  Binding the scheme_type
    py::enum_<scheme_type>(m, "scheme_type")
        .value("BFV", scheme_type::bfv)
        .value("CKKS", scheme_type::ckks)
        .export_values();
    
    // Binding the sec_level_type
    py::enum_<sec_level_type>(m, "sec_level_type")
        .value("none", sec_level_type::none)
        .value("tc128", sec_level_type::tc128)
        .value("tc192", sec_level_type::tc192)
        .value("tc256", sec_level_type::tc256)
        .export_values();

    // Bind EncryptionParameters
    py::class_<EncryptionParameters>(m, "EncryptionParameters")
        .def(py::init<scheme_type>())
        .def("set_poly_modulus_degree", &EncryptionParameters::set_poly_modulus_degree)
        .def("set_coeff_modulus", &EncryptionParameters::set_coeff_modulus)
        .def("set_plain_modulus", static_cast<void (EncryptionParameters::*)(std::uint64_t)>(&EncryptionParameters::set_plain_modulus))
        .def("scheme", &EncryptionParameters::scheme)
        .def("poly_modulus_degree", &EncryptionParameters::poly_modulus_degree)
        .def("coeff_modulus", &EncryptionParameters::coeff_modulus)
        .def("plain_modulus", &EncryptionParameters::plain_modulus); 

    // Binding Modulus
    py::class_<seal::Modulus>(m, "Modulus")
        .def(py::init<std::uint64_t>())
        .def("value", &seal::Modulus::value);

    // Binding CoeffModulus
    py::module coeff_mod = m.def_submodule("CoeffModulus");
    coeff_mod.def("BFVDefault", &CoeffModulus::BFVDefault);
    
    // Binding SEALContext
    py::class_<SEALContext, std::shared_ptr<SEALContext>>(m, "SEALContext")
        .def(py::init<const EncryptionParameters &>())
        .def("key_context_data", &SEALContext::key_context_data);
    
    // Binding Key Types
    py::class_<PublicKey>(m, "PublicKey")
        .def(py::init<>());
    py::class_<SecretKey>(m, "SecretKey")
        .def(py::init<>());
    py::class_<RelinKeys>(m, "RelinKeys")
        .def(py::init<>());
    py::class_<GaloisKeys>(m, "GaloisKeys")
        .def(py::init<>());

    // Binding public_key, secret_key, relin_keys and galois_keys 
    py::class_<KeyGenerator>(m, "KeyGenerator")
        .def(py::init<const SEALContext &>())
        .def("secret_key", &KeyGenerator::secret_key)
        .def("create_public_key", [](KeyGenerator &kgen) {
                 PublicKey pk;
                 kgen.create_public_key(pk);
                return pk;
            })
        .def("create_relin_keys", [](KeyGenerator &kgen) {
            RelinKeys rk;
            kgen.create_relin_keys(rk);
            return rk; // return relinearization
           })
       .def("create_galois_keys", [](KeyGenerator &kgen) {
            GaloisKeys gk;
            kgen.create_galois_keys(gk);
            return gk; // return galois keys
           });
    
    // Binding Plaintext
    py::class_<seal::Plaintext>(m, "Plaintext")
        .def(py::init<>())
        .def(py::init<const std::string &>())  // Initialize from hex string
        .def("to_string", &seal::Plaintext::to_string)
        .def("coeff_count", &seal::Plaintext::coeff_count)
        .def("is_zero", &seal::Plaintext::is_zero)
        .def("resize", &seal::Plaintext::resize)
        .def("data", py::overload_cast<>(&seal::Plaintext::data, py::const_)) //fixed
        .def("__str__", &seal::Plaintext::to_string)
        .def("save", [](const seal::Plaintext &pt, const std::string &path) {
            std::ofstream out(path, std::ios::binary);
            if (!out) throw std::runtime_error("Failed to open file for saving.");
            pt.save(static_cast<std::ostream&>(out));
            out.close();
        })
        .def("load", [](seal::Plaintext &pt, const seal::SEALContext &context, const std::string &filename) {
            std::ifstream in(filename, std::ios::binary);
            if (!in) {
               throw std::runtime_error("Failed to open file for loading Plaintext.");
           }
           pt.load(context, in);  
           in.close();
        });

    // Binding Ciphertext
    py::class_<seal::Ciphertext>(m, "Ciphertext")
        .def(py::init<>())
        .def(py::init<const seal::Ciphertext &>())
        .def("resize", [](seal::Ciphertext &ct, size_t cap, size_t degree) {
            ct.resize(cap, degree);
        })
        .def("size", &seal::Ciphertext::size)
        .def("poly_modulus_degree", &seal::Ciphertext::poly_modulus_degree)
        .def("is_ntt_form", &seal::Ciphertext::is_ntt_form)
        .def("scale", &seal::Ciphertext::scale)
        .def("__str__", [](const seal::Ciphertext &ct) {
            std::ostringstream oss;
            oss << "Ciphertext(size=" << ct.size()
                << ", poly_modulus_degree=" << ct.poly_modulus_degree() << ")";
            return oss.str();
        });



        
        
    
    // Binding Encryptor
    py::class_<Encryptor>(m, "Encryptor")
        .def(py::init<const SEALContext &, const seal::PublicKey &>())
        .def(py::init<const SEALContext &, const seal::SecretKey &>())
        .def("encrypt", [](Encryptor &enc, const seal::Plaintext &plain_text){
            seal::Ciphertext cipher_text;
            enc.encrypt(plain_text, cipher_text);
            return cipher_text; // return ciphertext
        });
    
        
        

        
    
}
